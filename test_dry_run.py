import sys

try:
    from data_loader import load_config, load_data
    from model import MLP
    from trainer import Trainer
    from visualizer import create_network_svg
    import numpy as np

    config = load_config()
    X, y, names, feature_names, target_class_names = load_data(config)
    print("Data Load OK")
    
    model = MLP(config['architecture'])
    trainer = Trainer(model, lr=0.01, loss_type='CrossEntropy')
    
    X_batch = X[0:2]
    y_batch = y[0:2]
    loss, current_weights, avg_deltas = trainer.step(X_batch, y_batch)
    print("Trainer Step OK, Loss:", loss)
    
    fig1 = create_network_svg(config['architecture'], current_weights, activations=model.last_activations, is_delta=False, target_class_names=target_class_names)
    fig2 = create_network_svg(config['architecture'], avg_deltas, is_delta=True, target_class_names=target_class_names)
    print("Visualizer SVG OK", len(fig1), len(fig2))

except Exception as e:
    print(f"Error occurred: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
