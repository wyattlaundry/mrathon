# mrathon
Multi-Rate Analyzer for Transient-Hybrids over Networks

# Network Synthesis
Constructing the network is model-independent, where devices are assigned to nodes, and lines are branched between nodes with pre-designated wiring configurations.

    # Lines
    LINE = FDLineModel.read('LineModel.json')
    
    # Nodes (Bus Bars)
    NODES = SimNode(GENA), SimNode(GENB)
    
    # Graph
    GRAPH = SimGraph(NODES)
    
    # Incidence
    GRAPH.connect(NODES[0], NODES[1], LINE)
