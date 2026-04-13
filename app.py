import streamlit as st
import time
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from data_loader import load_config, load_data
from model import MLP
from trainer import Trainer
from visualizer import create_network_plot

st.set_page_config(layout="wide", page_title="MLP Visualizer")

# Session State Initialization
if 'initialized' not in st.session_state:
    config = load_config()
    st.session_state.config = config
    X, y, names, feature_names, target_class_names = load_data(config)
    
    import numpy as np
    test_idx = np.random.randint(0, len(X))
    
    st.session_state.X_test = X[test_idx:test_idx+1]
    st.session_state.y_test = y[test_idx:test_idx+1]
    
    names_list = names.tolist() if hasattr(names, 'tolist') else list(names)
    st.session_state.name_test = names_list[test_idx]
    
    st.session_state.X = np.delete(X, test_idx, axis=0)
    st.session_state.y = np.delete(y, test_idx, axis=0)
    st.session_state.names = names_list[:test_idx] + names_list[test_idx+1:]
    
    st.session_state.feature_names = feature_names
    st.session_state.target_class_names = target_class_names
    
    st.session_state.model = MLP(config['architecture'])
    st.session_state.trainer = Trainer(st.session_state.model, lr=0.01, loss_type='CrossEntropy')
    
    st.session_state.current_sample_index = 0
    st.session_state.is_playing = False
    st.session_state.mode = 'Batch'
    st.session_state.loss_type = 'CrossEntropy'
    st.session_state.lr = 0.2
    
    st.session_state.trainer = Trainer(st.session_state.model, lr=0.2, loss_type='CrossEntropy')
    
    st.session_state.initialized = True

def reset_model():
    st.session_state.model = MLP(st.session_state.config['architecture'])
    st.session_state.trainer = Trainer(st.session_state.model, lr=st.session_state.lr, loss_type=st.session_state.loss_type)
    st.session_state.current_sample_index = 0

# --- Sidebar ---
st.sidebar.title("Controls")

col1, col2, col3 = st.sidebar.columns(3)
with col1:
    if st.sidebar.button("Play/Pause"):
        st.session_state.is_playing = not st.session_state.is_playing
with col2:
    if st.sidebar.button("Step"):
        st.session_state.is_playing = False
        st.session_state.step_flag = True
    else:
        st.session_state.step_flag = False
with col3:
    if st.sidebar.button("Reset"):
        reset_model()

st.session_state.playback_speed = st.sidebar.slider("Playback Speed (sec delay)", 0.0, 2.0, 0.5)

lr = st.sidebar.slider("Learning Rate", 0.001, 1.0, 0.2, step=0.005)
if lr != st.session_state.lr:
    st.session_state.lr = lr
    st.session_state.trainer.lr = lr
    for param_group in st.session_state.trainer.optimizer.param_groups:
        param_group['lr'] = lr

loss_type = st.sidebar.selectbox("Loss Function", ["CrossEntropy", "MSE", "MAE"], index=["CrossEntropy", "MSE", "MAE"].index(st.session_state.loss_type))
if loss_type != st.session_state.loss_type:
    st.session_state.loss_type = loss_type
    st.session_state.trainer.set_loss(loss_type)

mode = st.sidebar.selectbox("Optimization Mode", ["Batch", "SGD"], index=["Batch", "SGD"].index(st.session_state.mode))
if mode != st.session_state.mode:
    st.session_state.mode = mode

def step_forward():
    st.session_state.showing_test = False
    
    idx = st.session_state.current_sample_index
    st.session_state.last_evaluated_idx = idx

    if st.session_state.mode == 'SGD':
        X_batch = st.session_state.X[idx:idx+1]
        y_batch = st.session_state.y[idx:idx+1]
        st.session_state.trainer.step(X_batch, y_batch)
        st.session_state.current_sample_index = (idx + 1) % len(st.session_state.X)
    else:
        # Batch Mode
        if st.session_state.is_playing:
            # Play mode: train synchronously
            st.session_state.model.train()
            X_batch = st.session_state.X
            y_batch = st.session_state.y
            st.session_state.trainer.step(X_batch, y_batch)
            st.session_state.current_sample_index = 0
        else:
            # Step mode: pass one sample visually
            import torch
            st.session_state.model.eval()
            _ = st.session_state.model(
                torch.tensor(st.session_state.X[idx:idx+1], dtype=torch.float32), 
                record_activations=True
            )
            st.session_state.current_sample_index = (idx + 1) % len(st.session_state.X)
            
            # Train at the end of processing the batch
            if st.session_state.current_sample_index == 0:
                st.session_state.model.train()
                X_batch = st.session_state.X
                y_batch = st.session_state.y
                st.session_state.trainer.step(X_batch, y_batch)

if st.session_state.step_flag:
    step_forward()

# --- UI Layout ---
left_panel, canvas_right = st.columns([1, 3])

with left_panel:
    st.subheader("Data Feed")
    data_feed_ph = st.empty()
    
    st.markdown("---")
    st.subheader("Test Evaluation")
    st.info(f"Left-Out: **{st.session_state.get('name_test', 'None')}**")
    if st.button("Test on Left-Out Sample"):
        st.session_state.showing_test = True
        st.session_state.is_playing = False
        import torch
        st.session_state.model.eval()
        _ = st.session_state.model(
            torch.tensor(st.session_state.X_test, dtype=torch.float32), 
            record_activations=True
        )

with canvas_right:
    iteration_ph = st.empty()
    
    st.markdown("<h4 style='margin-bottom: -15px;'>Weights & Activations</h4>", unsafe_allow_html=True)
    fig1_ph = st.empty()
    
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    st.markdown("<h4 style='margin-bottom: -15px;'>Weight Deltas</h4>", unsafe_allow_html=True)
    fig2_ph = st.empty()

st.markdown("---")
st.subheader("Loss Curve")
loss_ph = st.empty()

def update_ui():
    iteration_ph.markdown(f"#### Batch / Iteration #: **{len(st.session_state.trainer.loss_history)}**")
    
    is_test = st.session_state.get('showing_test', False)
    
    if is_test:
        features_val = st.session_state.X_test[0].copy()
        current_name = f"{st.session_state.name_test} (TEST)"
        target_val = st.session_state.y_test[0]
    else:
        idx = st.session_state.get('last_evaluated_idx', st.session_state.current_sample_index)
        features_val = st.session_state.X[idx].copy()
        current_name = st.session_state.names[idx]
        target_val = st.session_state.y[idx]
    
    # Data feed update
    df_features = pd.DataFrame({"Feature": st.session_state.feature_names, "Value": features_val})
    if "legs" in st.session_state.feature_names:
        leg_idx = st.session_state.feature_names.index("legs")
        df_features.loc[leg_idx, "Value"] = int(round(df_features.loc[leg_idx, "Value"] * 8))
    
    original_target_val = target_val + 1  # Since we 0-indexed it from 1-7
    target_name = st.session_state.target_class_names[target_val] if target_val < len(st.session_state.target_class_names) else "Unknown"
    
    with data_feed_ph.container():
        if st.session_state.mode == 'Batch' and st.session_state.is_playing and not is_test:
            st.info("Batch Mode: Processing full epoch.")
        else:
            if is_test:
                st.warning("Viewing Test Sample Activations.")
            st.write(f"**Sample**: {current_name}")
            st.dataframe(df_features, hide_index=True)
            st.write(f"**Target Class**: {original_target_val} ({target_name})")

    # Weights update via SVG
    from visualizer import create_network_svg
    weights = st.session_state.model.get_weights()
    activations = getattr(st.session_state.model, 'last_activations', None)
    svg1 = create_network_svg(
        st.session_state.config['architecture'], 
        weights, 
        activations=activations, 
        is_delta=False,
        target_class_names=st.session_state.target_class_names,
        feature_names=st.session_state.feature_names
    )
    fig1_ph.markdown(svg1, unsafe_allow_html=True)

    # Deltas update via SVG
    trainer = st.session_state.trainer
    avg_deltas = []
    if len(trainer.delta_history) > 0:
        for l_idx in range(len(trainer.delta_history[-1])):
            avg_deltas.append(np.mean([d[l_idx] for d in trainer.delta_history], axis=0))
    svg2 = create_network_svg(
        st.session_state.config['architecture'], 
        avg_deltas, 
        activations=None, 
        is_delta=True,
        target_class_names=st.session_state.target_class_names,
        global_max_delta=trainer.global_max_delta
    )
    fig2_ph.markdown(svg2, unsafe_allow_html=True)

    # Loss update (using line_chart avoids iframe rebuilds)
    if len(st.session_state.trainer.loss_history) > 0:
        loss_df = pd.DataFrame({"Step": np.arange(1, len(st.session_state.trainer.loss_history) + 1), "Loss": st.session_state.trainer.loss_history})
        loss_ph.line_chart(loss_df.set_index("Step"))
    else:
        loss_ph.write("No training data yet.")

# Draw initial state natively without infinite looping initially
if not st.session_state.is_playing:
    update_ui()

if st.button("Take Snapshot"):
    st.success("You can right-click the graph to save the SVG!")

if st.session_state.is_playing:
    while st.session_state.is_playing:
        step_forward()
        update_ui()
        time.sleep(st.session_state.playback_speed)
