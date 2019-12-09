if __name__ == "__main__":
    import ipdb
    from morphx.processing import meshs, graphs, hybrids, visualize

    mh = meshs.load_mesh_gt('~/loc_Bachelorarbeit/GT/mesh/sso_2734465.pkl')
    local_bfs = graphs.local_bfs_dist(mh.graph(), 20, 10000)
    full_pc = hybrids.extract_cloud_subset(mh, local_bfs)

    mc = hybrids.extract_mesh_subset(mh, local_bfs)
    pc = meshs.sample_mesh_poisson_disk(mc, 5000)

    visualize.visualize_parallel([full_pc], [pc])
