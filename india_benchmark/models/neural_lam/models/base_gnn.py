from typing import List
import torch
import torch.nn as nn
from india_benchmark.models.neural_lam import utils
from india_benchmark.models.neural_lam.interaction_net import InteractionNet


class BaseGNN(nn.Module):
    """
    Base class for Graph Neural Network models.
    """
    def __init__(
        self,
        # data related
        img_size: List[int],
        variables: List[str],
        n_input_steps: int,
        # model architecture
        graph_dir_path: str,
        hidden_dim: int,
        hidden_layers: int,
    ):
        super().__init__()
        self.img_size = img_size
        self.num_grid_nodes = img_size[0] * img_size[1]
        self.grid_dim = n_input_steps * len(variables)
        self.grid_output_dim = len(variables)
        
        # Load graph with static features
        # NOTE: (IMPORTANT!) mesh nodes MUST have the first num_mesh_nodes indices
        self.hierarchical, graph_ldict = utils.load_graph(
            graph_dir_path=graph_dir_path
        )
        for name, attr_value in graph_ldict.items():
            # Make BufferLists module members and register tensors as buffers
            if isinstance(attr_value, torch.Tensor):
                self.register_buffer(name, attr_value, persistent=False)
            else:
                setattr(self, name, attr_value)
        
        # Specify dimensions of data
        self.num_mesh_nodes, _ = self.get_num_mesh()
        utils.rank_zero_print(
            f"Loaded graph with {self.num_grid_nodes + self.num_mesh_nodes} "
            f"nodes ({self.num_grid_nodes} grid, {self.num_mesh_nodes} mesh)"
        )

        # grid_dim from data + static
        self.g2m_edges, g2m_dim = self.g2m_features.shape
        self.m2g_edges, m2g_dim = self.m2g_features.shape

        # Define sub-models
        # Feature embedders for grid
        self.mlp_blueprint_end = [hidden_dim] * (hidden_layers + 1)
        self.grid_embedder = utils.make_mlp(
            [self.grid_dim] + self.mlp_blueprint_end
        )
        self.g2m_embedder = utils.make_mlp([g2m_dim] + self.mlp_blueprint_end)
        self.m2g_embedder = utils.make_mlp([m2g_dim] + self.mlp_blueprint_end)

        # GNNs
        # encoder
        self.g2m_gnn = InteractionNet(
            self.g2m_edge_index,
            hidden_dim,
            hidden_layers=hidden_layers,
            update_edges=False,
        )
        self.encoding_grid_mlp = utils.make_mlp(
            [hidden_dim] + self.mlp_blueprint_end
        )

        # decoder
        self.m2g_gnn = InteractionNet(
            self.m2g_edge_index,
            hidden_dim,
            hidden_layers=hidden_layers,
            update_edges=False,
        )

        # Output mapping (hidden_dim -> output_dim)
        self.output_map = utils.make_mlp(
            [hidden_dim] * (hidden_layers + 1)
            + [self.grid_output_dim],
            layer_norm=False,
        )  # No layer norm on this one
    
    def get_num_mesh(self):
        """
        Compute number of mesh nodes from loaded features,
        and number of mesh nodes that should be ignored in encoding/decoding
        """
        raise NotImplementedError("get_num_mesh not implemented")

    def embedd_mesh_nodes(self):
        """
        Embed static mesh features
        Returns tensor of shape (num_mesh_nodes, d_h)
        """
        raise NotImplementedError("embedd_mesh_nodes not implemented")

    def process_step(self, mesh_rep):
        """
        Process step of embedd-process-decode framework
        Processes the representation on the mesh, possible in multiple steps

        mesh_rep: has shape (B, num_mesh_nodes, d_h)
        Returns mesh_rep: (B, num_mesh_nodes, d_h)
        """
        raise NotImplementedError("process_step not implemented")
    
    def grid_to_nodes(self, x: torch.Tensor):
        # x: (B, n_steps, c, h, w)
        b, n_steps, c, _, _ = x.shape
        return x.view(b, n_steps, c, -1).permute(0, 1, 3, 2).contiguous() # (B, n_steps, N, c)
    
    def nodes_to_grid(self, x):
        # x: (B, N, c)
        b, _, c = x.shape
        return x.permute(0, 2, 1).view(b, c, *self.img_size)
    
    @staticmethod
    def expand_to_batch(x, batch_size):
        """
        Expand tensor with initial batch dimension
        """
        return x.unsqueeze(0).expand(batch_size, -1, -1)
    
    def forward(self, prev_states):
        """
        Step state one step ahead
        prev_states: (B, n_steps, c, h, w)
        """
        batch_size, n_steps = prev_states.shape[:2]
        prev_states = self.grid_to_nodes(prev_states) # (B, n_steps, num_grid_nodes, c)
        # concat all previous states over the channel dimension
        grid_features = torch.cat([prev_states[:, i] for i in range(n_steps)], dim=-1) # (B, num_grid_nodes, n_steps * c)

        # Embed all features
        grid_emb = self.grid_embedder(grid_features)  # (B, num_grid_nodes, d_h)
        g2m_emb = self.g2m_embedder(self.g2m_features)  # (M_g2m, d_h)
        m2g_emb = self.m2g_embedder(self.m2g_features)  # (M_m2g, d_h)
        mesh_emb = self.embedd_mesh_nodes()

        # Map from grid to mesh
        mesh_emb_expanded = self.expand_to_batch(
            mesh_emb, batch_size
        )  # (B, num_mesh_nodes, d_h)
        g2m_emb_expanded = self.expand_to_batch(g2m_emb, batch_size)

        # This also splits representation into grid and mesh
        mesh_rep = self.g2m_gnn(
            grid_emb, mesh_emb_expanded, g2m_emb_expanded
        )  # (B, num_mesh_nodes, d_h)
        # Also MLP with residual for grid representation
        grid_rep = grid_emb + self.encoding_grid_mlp(
            grid_emb
        )  # (B, num_grid_nodes, d_h)

        # Run processor step
        mesh_rep = self.process_step(mesh_rep)

        # Map back from mesh to grid
        m2g_emb_expanded = self.expand_to_batch(m2g_emb, batch_size)
        grid_rep = self.m2g_gnn(
            mesh_rep, grid_rep, m2g_emb_expanded
        )  # (B, num_grid_nodes, d_h)

        # Map to output dimension
        pred_delta_mean = self.output_map(grid_rep)  # (B, num_grid_nodes, d_grid_out)
        return self.nodes_to_grid(pred_delta_mean)