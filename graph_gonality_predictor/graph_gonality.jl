using Graphs
using GraphNeuralNetworks
using Flux
using Zygote
using Random
using Statistics # For mean
using ChipFiring # Added based on your latest code
using Plots


"""
Generates a dataset of random graphs for training.

Generates 1/3 Erdos-Renyi graphs, 1/3 Watts-Strogatz, 1/3 Barabasi-Albert graphs.

Picks a random number of edges and vertices.
"""
function generate_graph_dataset(num_graphs::Int, max_nodes::Int = 20)
    graphs = []
    # Use a while loop to ensure we get exactly num_graphs
    while length(graphs) < num_graphs
        num_v = rand(5:max_nodes)
        
        # Randomly choose a graph type
        graph_type = rand(1:3)
        
        local g # Need to declare g here to be in scope

        if graph_type == 1
            # 1. Erdos-Renyi (Random)
            num_e = rand((num_v - 1):((num_v * (num_v - 1)) ÷ 2))
            g = erdos_renyi(num_v, num_e)
            
        elseif graph_type == 2
            # 2. Watts-Strogatz (Small-world)
            # k-neighbors, e.g., 2, 4, 6 (must be even)
            k = rand(2:2:min(num_v - 2, 6)) 
            beta = rand(0.05:0.05:0.3)      # Rewiring probability
            g = watts_strogatz(num_v, k, beta)

        elseif graph_type == 3
            # 3. Barabasi-Albert (Scale-free)
            # k = number of edges to attach from a new node
            k = rand(1:min(num_v - 1, 4)) # e.g., 1, 2, 3, 4
            g = barabasi_albert(num_v, k)
        end
        
        # Ensure graph is connected, as gonality is often defined for connected graphs
        if is_connected(g)
            push!(graphs, g)
        end
    end
    return graphs
end

"""
Converts a Graphs.jl graph into a GNNGraph with node features.
The GNN needs numerical features for its nodes, and we choose to just use the node degree.
"""
function featurize_graph(g::AbstractGraph, feature_dim::Int)
    num_nodes = nv(g)
    
    # --- Node Features ---
    # We'll use one-hot encoding of the node degree as features.
    # The feature_dim is fixed.
    
    node_features = zeros(Float32, feature_dim, num_nodes)
    
    for v in 1:num_nodes
        d = degree(g, v)
        # We map degree 'd' to index 'd + 1'
        if d + 1 <= feature_dim
            node_features[d + 1, v] = 1.0
        else
            # If a node's degree is too high (e.g., in a test graph),
            # we'll just cap it at the max feature dimension.
            node_features[feature_dim, v] = 1.0
        end
    end

    return GNNGraph(g, ndata = node_features)
end

"""
Creates the full labeled dataset.
This is the "taxing" part because it calls compute_gonality.

FIXED: Passes the `feature_dim` to featurize_graph.
FIXED: Uses the new compute_graph_gonality_from_graph function.
"""
function create_labeled_dataset(graphs::Vector, feature_dim::Int)
    dataset = []
    for g in graphs
        # 1. Featurize the graph for GNN input (X)
        gnn_graph = featurize_graph(g, feature_dim)
        
        # 2. Compute the expensive ground truth label (y)
        gonality_val = Float32(compute_graph_gonality_from_graph(g))
        
        # Skip graphs where gonality computation failed
        if gonality_val > 0
            push!(dataset, (gnn_graph, gonality_val))
        end
    end
    return dataset
end

"""
Splits a dataset into training and validation sets.
"""
function train_val_split(dataset::Vector, split_ratio::Float64 = 0.8)
    if isempty(dataset)
        return [], []
    end
    shuffled_dataset = dataset[shuffle(1:end)]
    split_index = floor(Int, split_ratio * length(shuffled_dataset))
    train_set = shuffled_dataset[1:split_index]
    val_set = shuffled_dataset[(split_index + 1):end]
    return train_set, val_set
end


# --- 3. Model Definition (GNN) ---

"""
Defines the Graph Neural Network model architecture.
"""
function build_model(input_feature_dim::Int)
    # We define a GNN with:
    # 1. A Graph Convolution (GCN) layer
    # 2. A pooling layer to aggregate node info into a single graph-level vector
    # 3. A dense layer to predict the final number (gonality)
    
    hidden_dim = 64 # Number of features in the hidden layer
    
    model = GNNChain(
        GCNConv(input_feature_dim => hidden_dim, leakyrelu),
        (x -> mean(x, dims=2)), # Aggregates all node features into one vector (global pooling)
        Dense(hidden_dim => 32, leakyrelu), 
        Dense(32 => 1)
    )
    
    return model
end

# --- 4. Training ---

"""
The main training loop.

"""
function train_gonality_model(model, train_set, val_set, epochs::Int = 50)
    # Loss: MSE
    loss(m, gnn_g, y) = Flux.mse(m(gnn_g, gnn_g.ndata.x), y)
    
    # Optimizer: Adam
    optimizer = Flux.setup(Adam(0.001), model)
    
    train_losses = Float32[]
    val_losses = Float32[]

    println("Starting training...")
    
    if isempty(train_set)
        println("ERROR: Training dataset is empty. Cannot train.")
        return model, (train_losses, val_losses)
    end

    for epoch in 1:epochs
        # --- Training Phase ---
        epoch_train_loss = 0.0
        shuffled_train_set = train_set[shuffle(1:end)]
        
        for (gnn_graph, y) in shuffled_train_set
            # Calculate loss and gradients
            l, grads = Flux.withgradient(model) do m
                # We need to reshape y to (1,1) for Flux.mse
                y_reshaped = reshape([y], 1, 1)
                # FIXED: Pass the tracked model `m` to the loss function
                loss(m, gnn_graph, y_reshaped)
            end
            
            # Update the model's parameters
            Flux.update!(optimizer, model, grads[1])
            epoch_train_loss += l
        end
        
        # --- Validation Phase ---
        epoch_val_loss = 0.0
        if !isempty(val_set)
            for (gnn_graph, y) in val_set
                # Reshape y
                y_reshaped = reshape([y], 1, 1)
                # Calculate loss without gradients
                epoch_val_loss += loss(model, gnn_graph, y_reshaped)
            end
        end

        # --- Reporting ---
        avg_train_loss = epoch_train_loss / length(train_set)
        avg_val_loss = isempty(val_set) ? 0.0 : (epoch_val_loss / length(val_set))
        
        push!(train_losses, avg_train_loss)
        push!(val_losses, avg_val_loss)

        if epoch % 10 == 0
            if isempty(val_set)
                println("Epoch: $epoch | Train Loss: $avg_train_loss")
            else
                println("Epoch: $epoch | Train Loss: $avg_train_loss | Validation Loss: $avg_val_loss")
            end
        end
    end
    
    println("Training complete.")

    return model, (train_losses, val_losses)
end

# --- 5. Prediction ---

"""
Uses the trained model to predict the gonality of a new graph.

FIXED: Accepts `feature_dim` to featurize the graph correctly.
"""
function predict_gonality(model, g::AbstractGraph, feature_dim::Int)
    # Featurize the new graph in the same way as the training data
    gnn_graph = featurize_graph(g, feature_dim)
    
    # Get the model's prediction
    # The output will be a 1x1 matrix, so we extract the single value
    predicted_value = model(gnn_graph, gnn_graph.ndata.x)[1, 1]
    
    return predicted_value
end

"""
Calculates the 0/1 accuracy of the model on a given dataset.
0/1 loss means: is the rounded prediction *exactly* equal to the true value?
"""
function test_model_accuracy(model, dataset::Vector, feature_dim::Int)
    if isempty(dataset)
        println("Cannot calculate accuracy on empty dataset.")
        return 0.0
    end

    num_correct = 0
    for (gnn_graph, y_true) in dataset
        # Get the raw prediction (e.g., 3.1 or 2.8)
        y_pred_raw = model(gnn_graph, gnn_graph.ndata.x)[1, 1]
        
        # Round to the nearest integer
        y_pred_rounded = round(Int, y_pred_raw)
        
        # Check if it matches the true value
        if y_pred_rounded == round(Int, y_true)
            num_correct += 1
        end
    end
    
    # Return the accuracy (e.g., 90.0%)
    return (num_correct / length(dataset)) * 100.0
end


# --- 6. Main Execution ---
function main()
    println("--- Gonality Predictor ML Pipeline ---")
    
    # --- Data Generation ---
    # Define max_nodes and feature_dim *once*
    # If max_nodes is 10, the max possible degree is 9.
    # Our feature vector needs to encode degrees 0 through 9 (10 slots).
    # So, input_feature_dim = max_nodes_in_dataset.
    max_nodes_in_dataset = 12
    input_feature_dim = max_nodes_in_dataset 
    
    # Generate a larger dataset to split
    total_graphs = 10000 # e.g., 200 for train, 50 for val
    println("1. Generating $total_graphs random graphs for training (max_nodes = $max_nodes_in_dataset)...")
    raw_graphs = generate_graph_dataset(total_graphs, max_nodes_in_dataset)
    println("Generated $(length(raw_graphs)) connected graphs.")
    
    println("2. Computing expensive labels (gonality) and featurizing (feature_dim = $input_feature_dim)...")
    # This is the slow step
    # FIXED: Pass input_feature_dim to create the dataset
    dataset = create_labeled_dataset(raw_graphs, input_feature_dim)
    println("Dataset created with $(length(dataset)) samples.")

    # --- FIXED: Split Dataset ---
    (train_set, val_set) = train_val_split(dataset, 0.8) # 80% train, 20% val
    println("Split dataset into $(length(train_set)) training samples and $(length(val_set)) validation samples.")

    # --- Model Training ---
    # The input dimension must match our feature engineering
    model = build_model(input_feature_dim)
    
    println("3. Training GNN model...")


    (trained_model, (train_losses, val_losses)) = train_gonality_model(model, train_set, val_set, 100)
    
    # Plotting
    if !isempty(train_losses) && !isempty(val_losses)
        println("\n4. Generating loss plot...")
        epochs_axis = 1:length(train_losses)
        p = plot(epochs_axis, 
                 [train_losses, val_losses], 
                 label=["Training Loss" "Validation Loss"],
                 xlabel="Epoch",
                 ylabel="Loss (MSE)",
                 title="GNN Training and Validation Loss",
                 legend=:topright,
                 yaxis=:log) # Use log scale for loss
        
        # Save the plot
        plot_filename = "gonality_loss_plot.png"
        savefig(p, plot_filename)
        println("Loss plot saved to $plot_filename")
    else
        println("\n4. Skipping plot generation (no training data).")
    end

    # --- FIXED: Report 0/1 Task Loss (Accuracy) ---
    println("\n--- Final Model Performance ---")
    val_accuracy = test_model_accuracy(trained_model, val_set, input_feature_dim)
    println("Final Validation Accuracy (0/1 Loss): $(round(val_accuracy, digits=2))%")
end

# Run the main function
main()