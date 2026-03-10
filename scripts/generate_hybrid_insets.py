import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import os

# Set plotting style for Nature-like aesthetics
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.linewidth'] = 0.5

# Create output dir
OUTPUT_DIR = 'docs/mypaper/figures/insets'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_inset(name):
    plt.savefig(os.path.join(OUTPUT_DIR, f'{name}.pdf'), 
                bbox_inches='tight', 
                pad_inches=0, 
                transparent=True)
    print(f"Generated {name}.pdf")
    plt.close()

def generate_reward_inset():
    try:
        mat = scipy.io.loadmat('models/RRSACPSO/rrsacpso_perparticle_log.mat')
        # Structure is logData[0,0]['episodeRewards']
        rewards = mat['logData'][0,0]['episodeRewards'].flatten()
        
        # Smooth the rewards for better visual flow
        window = 20
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        
        fig, ax = plt.subplots(figsize=(2, 1))
        # Fill area under curve
        ax.fill_between(range(len(smoothed)), smoothed, alpha=0.2, color='#E91E63')
        ax.plot(smoothed, color='#E91E63', linewidth=1.2)
        
        ax.axis('off')
        save_inset('reward_inset')
    except Exception as e:
        print(f"Error generating reward inset: {e}")

def generate_features_inset():
    # Generate a realistic-looking temporal window of 9 features
    # (High variance at start, stabilizing over time)
    np.random.seed(42)
    time_steps = 30
    n_features = 9
    
    # Base pattern: some features are large, some small
    base = np.linspace(0.1, 0.9, n_features).reshape(1, n_features)
    # Add noise that decays
    noise = np.random.randn(time_steps, n_features) * np.linspace(0.5, 0.1, time_steps)[:, None]
    data = (base + noise).T
    
    fig, ax = plt.subplots(figsize=(2, 1))
    im = ax.imshow(data, aspect='auto', cmap='viridis', interpolation='nearest')
    
    ax.axis('off')
    # Add a thin border
    for spine in ax.spines.values():
        spine.set_visible(False)
        
    save_inset('features_inset')

def generate_transformer_inset():
    # Minimalist schematic of Transformer blocks
    fig, ax = plt.subplots(figsize=(2, 2))
    
    # Draw stacked blocks with slight offset for depth
    colors = ['#E3F2FD', '#BBDEFB', '#90CAF9']
    for i in range(3):
        # Attention block
        rect = plt.Rectangle((0.15, 0.2 + i*0.25), 0.7, 0.15, 
                            facecolor=colors[i], edgecolor='#1976D2', 
                            linewidth=0.8, alpha=0.9)
        ax.add_patch(rect)
        ax.text(0.5, 0.275 + i*0.25, 'Attention', ha='center', va='center', 
                fontsize=7, color='#0D47A1', fontweight='bold')
        
        # Mini arrows between blocks
        if i < 2:
            ax.annotate('', xy=(0.5, 0.45 + i*0.25), xytext=(0.5, 0.35 + i*0.25),
                       arrowprops=dict(arrowstyle='->', lw=0.5, color='#1976D2'))
            
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    save_inset('transformer_inset')

def generate_pso_update_inset():
    # A vector diagram showing Inertia, Cognitive, and Social components
    fig, ax = plt.subplots(figsize=(2, 2))
    
    # Origin (current position x_t)
    org = np.array([0, 0])
    
    # Vectors
    v_inertia = np.array([0.7, 0.2])
    v_cog = np.array([0.1, 0.8])
    v_social = np.array([0.4, 0.5])
    v_total = v_inertia + v_cog + v_social
    
    # Draw vectors with professional colors
    ax.quiver(*org, *v_inertia, color='#9E9E9E', scale=1, scale_units='xy', 
             angles='xy', width=0.03, label='Inertia')
    ax.quiver(*v_inertia, *v_cog, color='#4CAF50', scale=1, scale_units='xy', 
             angles='xy', width=0.03, label='Cognitive')
    ax.quiver(*(v_inertia + v_cog), *v_social, color='#2196F3', scale=1, scale_units='xy', 
             angles='xy', width=0.03, label='Social')
    ax.quiver(*org, *v_total, color='#E91E63', scale=1, scale_units='xy', 
             angles='xy', width=0.05, label='Resultant')
    
    # Text labels
    ax.text(0.3, 0.05, '$\omega v_t$', color='#757575', fontsize=8)
    ax.text(0.4, 0.85, '$c_1 r_1$', color='#388E3C', fontsize=8)
    ax.text(1.0, 0.8, '$c_2 r_2$', color='#1976D2', fontsize=8)
    ax.text(0.8, 0.4, '$v_{t+1}$', color='#C2185B', fontsize=9, fontweight='bold')
    
    # Particle dot
    ax.plot(0, 0, 'ko', markersize=4)
    
    ax.set_xlim(-0.1, 1.5)
    ax.set_ylim(-0.1, 1.8)
    ax.axis('off')
    save_inset('pso_update_inset')

if __name__ == "__main__":
    generate_reward_inset()
    generate_features_inset()
    generate_transformer_inset()
    generate_pso_update_inset()
