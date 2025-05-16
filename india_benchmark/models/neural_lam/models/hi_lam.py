from typing import List
from torch import nn
from india_benchmark.models.neural_lam import utils
from india_benchmark.models.neural_lam.interaction_net import InteractionNet
from india_benchmark.models.neural_lam.models.base_gnn import BaseGNN


class HiLAM(BaseGNN):
    """
    Base class for hierarchical graph models.
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
        processor_layers: int,
    ):
        super().__init__(
            img_size=img_size,
            variables=variables,
            n_input_steps=n_input_steps,
            graph_dir_path=graph_dir_path,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
        )

        # Track number of nodes, edges on each level
        # Flatten lists for efficient embedding
        self.num_levels = len(self.mesh_static_features)

        # Number of mesh nodes at each level
        self.level_mesh_sizes = [
            mesh_feat.shape[0] for mesh_feat in self.mesh_static_features
        ]  # Needs as python list for later

        # Print some useful info
        utils.rank_zero_print("Loaded hierarchical graph with structure:")
        for level_index, level_mesh_size in enumerate(self.level_mesh_sizes):
            same_level_edges = self.m2m_features[level_index].shape[0]
            utils.rank_zero_print(
                f"level {level_index} - {level_mesh_size} nodes, "
                f"{same_level_edges} same-level edges"
            )

            if level_index < (self.num_levels - 1):
                up_edges = self.mesh_up_features[level_index].shape[0]
                down_edges = self.mesh_down_features[level_index].shape[0]
                utils.rank_zero_print(f"  {level_index}<->{level_index + 1}")
                utils.rank_zero_print(
                    f" - {up_edges} up edges, {down_edges} down edges"
                )
        # Embedders
        # Assume all levels have same static feature dimensionality
        mesh_dim = self.mesh_static_features[0].shape[1]
        mesh_same_dim = self.m2m_features[0].shape[1]
        mesh_up_dim = self.mesh_up_features[0].shape[1]
        mesh_down_dim = self.mesh_down_features[0].shape[1]

        # Separate mesh node embedders for each level
        self.mesh_embedders = nn.ModuleList(
            [
                utils.make_mlp([mesh_dim] + self.mlp_blueprint_end)
                for _ in range(self.num_levels)
            ]
        )
        self.mesh_same_embedders = nn.ModuleList(
            [
                utils.make_mlp([mesh_same_dim] + self.mlp_blueprint_end)
                for _ in range(self.num_levels)
            ]
        )
        self.mesh_up_embedders = nn.ModuleList(
            [
                utils.make_mlp([mesh_up_dim] + self.mlp_blueprint_end)
                for _ in range(self.num_levels - 1)
            ]
        )
        self.mesh_down_embedders = nn.ModuleList(
            [
                utils.make_mlp([mesh_down_dim] + self.mlp_blueprint_end)
                for _ in range(self.num_levels - 1)
            ]
        )

        # Instantiate GNNs
        # Init GNNs
        self.mesh_init_gnns = nn.ModuleList(
            [
                InteractionNet(
                    edge_index,
                    hidden_dim,
                    hidden_layers=hidden_layers,
                )
                for edge_index in self.mesh_up_edge_index
            ]
        )

        # Read out GNNs
        self.mesh_read_gnns = nn.ModuleList(
            [
                InteractionNet(
                    edge_index,
                    hidden_dim,
                    hidden_layers=hidden_layers,
                    update_edges=False,
                )
                for edge_index in self.mesh_down_edge_index
            ]
        )
        
        # Make down GNNs, both for down edges and same level
        self.mesh_down_gnns = nn.ModuleList(
            [self.make_down_gnns(hidden_dim, hidden_layers) for _ in range(processor_layers)]
        )  # Nested lists (proc_steps, num_levels-1)
        self.mesh_down_same_gnns = nn.ModuleList(
            [self.make_same_gnns(hidden_dim, hidden_layers) for _ in range(processor_layers)]
        )  # Nested lists (proc_steps, num_levels)

        # Make up GNNs, both for up edges and same level
        self.mesh_up_gnns = nn.ModuleList(
            [self.make_up_gnns(hidden_dim, hidden_layers) for _ in range(processor_layers)]
        )  # Nested lists (proc_steps, num_levels-1)
        self.mesh_up_same_gnns = nn.ModuleList(
            [self.make_same_gnns(hidden_dim, hidden_layers) for _ in range(processor_layers)]
        )  # Nested lists (proc_steps, num_levels)

    def get_num_mesh(self):
        """
        Compute number of mesh nodes from loaded features,
        and number of mesh nodes that should be ignored in encoding/decoding
        """
        num_mesh_nodes = sum(
            node_feat.shape[0] for node_feat in self.mesh_static_features
        )
        num_mesh_nodes_ignore = (
            num_mesh_nodes - self.mesh_static_features[0].shape[0]
        )
        return num_mesh_nodes, num_mesh_nodes_ignore

    def embedd_mesh_nodes(self):
        """
        Embed static mesh features
        This embeds only bottom level, rest is done at beginning of
        processing step
        Returns tensor of shape (num_mesh_nodes[0], d_h)
        """
        return self.mesh_embedders[0](self.mesh_static_features[0])

    def process_step(self, mesh_rep):
        """
        Process step of embedd-process-decode framework
        Processes the representation on the mesh, possible in multiple steps

        mesh_rep: has shape (B, num_mesh_nodes, d_h)
        Returns mesh_rep: (B, num_mesh_nodes, d_h)
        """
        batch_size = mesh_rep.shape[0]

        # EMBED REMAINING MESH NODES (levels >= 1) -
        # Create list of mesh node representations for each level,
        # each of size (B, num_mesh_nodes[l], d_h)
        mesh_rep_levels = [mesh_rep] + [
            self.expand_to_batch(emb(node_static_features), batch_size)
            for emb, node_static_features in zip(
                list(self.mesh_embedders)[1:],
                list(self.mesh_static_features)[1:],
            )
        ]

        # - EMBED EDGES -
        # Embed edges, expand with batch dimension
        mesh_same_rep = [
            self.expand_to_batch(emb(edge_feat), batch_size)
            for emb, edge_feat in zip(
                self.mesh_same_embedders, self.m2m_features
            )
        ]
        mesh_up_rep = [
            self.expand_to_batch(emb(edge_feat), batch_size)
            for emb, edge_feat in zip(
                self.mesh_up_embedders, self.mesh_up_features
            )
        ]
        mesh_down_rep = [
            self.expand_to_batch(emb(edge_feat), batch_size)
            for emb, edge_feat in zip(
                self.mesh_down_embedders, self.mesh_down_features
            )
        ]

        # - MESH INIT. -
        # Let level_l go from 1 to L
        for level_l, gnn in enumerate(self.mesh_init_gnns, start=1):
            # Extract representations
            send_node_rep = mesh_rep_levels[
                level_l - 1
            ]  # (B, num_mesh_nodes[l-1], d_h)
            rec_node_rep = mesh_rep_levels[
                level_l
            ]  # (B, num_mesh_nodes[l], d_h)
            edge_rep = mesh_up_rep[level_l - 1]

            # Apply GNN
            new_node_rep, new_edge_rep = gnn(
                send_node_rep, rec_node_rep, edge_rep
            )

            # Update node and edge vectors in lists
            mesh_rep_levels[
                level_l
            ] = new_node_rep  # (B, num_mesh_nodes[l], d_h)
            mesh_up_rep[level_l - 1] = new_edge_rep  # (B, M_up[l-1], d_h)

        # - PROCESSOR -
        mesh_rep_levels, _, _, mesh_down_rep = self.hi_processor_step(
            mesh_rep_levels, mesh_same_rep, mesh_up_rep, mesh_down_rep
        )

        # - MESH READ OUT. -
        # Let level_l go from L-1 to 0
        for level_l, gnn in zip(
            range(self.num_levels - 2, -1, -1), reversed(self.mesh_read_gnns)
        ):
            # Extract representations
            send_node_rep = mesh_rep_levels[
                level_l + 1
            ]  # (B, num_mesh_nodes[l+1], d_h)
            rec_node_rep = mesh_rep_levels[
                level_l
            ]  # (B, num_mesh_nodes[l], d_h)
            edge_rep = mesh_down_rep[level_l]

            # Apply GNN
            new_node_rep = gnn(send_node_rep, rec_node_rep, edge_rep)

            # Update node and edge vectors in lists
            mesh_rep_levels[
                level_l
            ] = new_node_rep  # (B, num_mesh_nodes[l], d_h)

        # Return only bottom level representation
        return mesh_rep_levels[0]  # (B, num_mesh_nodes[0], d_h)
    
    def make_same_gnns(self, hidden_dim, hidden_layers):
        """
        Make intra-level GNNs.
        """
        return nn.ModuleList(
            [
                InteractionNet(
                    edge_index,
                    hidden_dim,
                    hidden_layers=hidden_layers,
                )
                for edge_index in self.m2m_edge_index
            ]
        )

    def make_up_gnns(self, hidden_dim, hidden_layers):
        """
        Make GNNs for processing steps up through the hierarchy.
        """
        return nn.ModuleList(
            [
                InteractionNet(
                    edge_index,
                    hidden_dim,
                    hidden_layers=hidden_layers,
                )
                for edge_index in self.mesh_up_edge_index
            ]
        )

    def make_down_gnns(self, hidden_dim, hidden_layers):
        """
        Make GNNs for processing steps down through the hierarchy.
        """
        return nn.ModuleList(
            [
                InteractionNet(
                    edge_index,
                    hidden_dim,
                    hidden_layers=hidden_layers,
                )
                for edge_index in self.mesh_down_edge_index
            ]
        )

    def mesh_down_step(
        self,
        mesh_rep_levels,
        mesh_same_rep,
        mesh_down_rep,
        down_gnns,
        same_gnns,
    ):
        """
        Run down-part of vertical processing, sequentially alternating between
        processing using down edges and same-level edges.
        """
        # Run same level processing on level L
        mesh_rep_levels[-1], mesh_same_rep[-1] = same_gnns[-1](
            mesh_rep_levels[-1], mesh_rep_levels[-1], mesh_same_rep[-1]
        )

        # Let level_l go from L-1 to 0
        for level_l, down_gnn, same_gnn in zip(
            range(self.num_levels - 2, -1, -1),
            reversed(down_gnns),
            reversed(same_gnns[:-1]),
        ):
            # Extract representations
            send_node_rep = mesh_rep_levels[
                level_l + 1
            ]  # (B, N_mesh[l+1], d_h)
            rec_node_rep = mesh_rep_levels[level_l]  # (B, N_mesh[l], d_h)
            down_edge_rep = mesh_down_rep[level_l]
            same_edge_rep = mesh_same_rep[level_l]

            # Apply down GNN
            new_node_rep, mesh_down_rep[level_l] = down_gnn(
                send_node_rep, rec_node_rep, down_edge_rep
            )

            # Run same level processing on level l
            mesh_rep_levels[level_l], mesh_same_rep[level_l] = same_gnn(
                new_node_rep, new_node_rep, same_edge_rep
            )
            # (B, N_mesh[l], d_h) and (B, M_same[l], d_h)

        return mesh_rep_levels, mesh_same_rep, mesh_down_rep

    def mesh_up_step(
        self, mesh_rep_levels, mesh_same_rep, mesh_up_rep, up_gnns, same_gnns
    ):
        """
        Run up-part of vertical processing, sequentially alternating between
        processing using up edges and same-level edges.
        """

        # Run same level processing on level 0
        mesh_rep_levels[0], mesh_same_rep[0] = same_gnns[0](
            mesh_rep_levels[0], mesh_rep_levels[0], mesh_same_rep[0]
        )

        # Let level_l go from 1 to L
        for level_l, (up_gnn, same_gnn) in enumerate(
            zip(up_gnns, same_gnns[1:]), start=1
        ):
            # Extract representations
            send_node_rep = mesh_rep_levels[
                level_l - 1
            ]  # (B, N_mesh[l-1], d_h)
            rec_node_rep = mesh_rep_levels[level_l]  # (B, N_mesh[l], d_h)
            up_edge_rep = mesh_up_rep[level_l - 1]
            same_edge_rep = mesh_same_rep[level_l]

            # Apply up GNN
            new_node_rep, mesh_up_rep[level_l - 1] = up_gnn(
                send_node_rep, rec_node_rep, up_edge_rep
            )
            # (B, N_mesh[l], d_h) and (B, M_up[l-1], d_h)

            # Run same level processing on level l
            mesh_rep_levels[level_l], mesh_same_rep[level_l] = same_gnn(
                new_node_rep, new_node_rep, same_edge_rep
            )
            # (B, N_mesh[l], d_h) and (B, M_same[l], d_h)

        return mesh_rep_levels, mesh_same_rep, mesh_up_rep

    def hi_processor_step(
        self, mesh_rep_levels, mesh_same_rep, mesh_up_rep, mesh_down_rep
    ):
        """
        Internal processor step of hierarchical graph models.
        Between mesh init and read out.

        Each input is list with representations, each with shape

        mesh_rep_levels: (B, N_mesh[l], d_h)
        mesh_same_rep: (B, M_same[l], d_h)
        mesh_up_rep: (B, M_up[l -> l+1], d_h)
        mesh_down_rep: (B, M_down[l <- l+1], d_h)

        Returns same lists
        """
        for down_gnns, down_same_gnns, up_gnns, up_same_gnns in zip(
            self.mesh_down_gnns,
            self.mesh_down_same_gnns,
            self.mesh_up_gnns,
            self.mesh_up_same_gnns,
        ):
            # Down
            mesh_rep_levels, mesh_same_rep, mesh_down_rep = self.mesh_down_step(
                mesh_rep_levels,
                mesh_same_rep,
                mesh_down_rep,
                down_gnns,
                down_same_gnns,
            )

            # Up
            mesh_rep_levels, mesh_same_rep, mesh_up_rep = self.mesh_up_step(
                mesh_rep_levels,
                mesh_same_rep,
                mesh_up_rep,
                up_gnns,
                up_same_gnns,
            )

        # NOTE: We return all, even though only down edges really are used
        # later
        return mesh_rep_levels, mesh_same_rep, mesh_up_rep, mesh_down_rep

# model = HiLAM(
#     img_size=[256, 256],
#     variables=[
#         "TMP",
#         "UGRD",
#         "VGRD",
#         "PRMSL",
#         "HGT925",
#         "HGT850",
#         "HGT700",
#         "HGT600",
#         "HGT500",
#         "HGT250",
#         "HGT50",
#     ],
#     n_input_steps=2,
#     graph_dir_path='/eagle/MDClimSim/tungnd/data/imdaa/hierarchical_graph',
#     hidden_dim=64,
#     hidden_layers=1,
#     processor_layers=4
# ).cuda()
# import torch
# prev_states = torch.randn(4, 2, 11, 256, 256).cuda()
# output = model(prev_states)
# print(output.shape)