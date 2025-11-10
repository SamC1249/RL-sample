"""
Visualization of Gravitational Field
Displays gravity gradient, celestial bodies, and field vectors across the grid world.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib import cm
from gravitational_env import GravitationalDynamicsEnv


def visualize_gravity_field(env: GravitationalDynamicsEnv, save_path: str = None):
    """
    Visualize the gravitational field with:
    - Heatmap of gravity magnitude
    - Quiver plot of gravity vectors
    - Celestial bodies and event horizons
    """
    # Get gravity field
    gravity_field = env.get_gravity_field()

    # Compute gravity magnitude
    gravity_magnitude = np.sqrt(gravity_field[:, :, 0]**2 + gravity_field[:, :, 1]**2)

    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # --- Plot 1: Gravity Magnitude Heatmap ---
    ax1 = axes[0]
    im1 = ax1.imshow(gravity_magnitude.T, origin='lower', cmap='hot', interpolation='bilinear')
    ax1.set_title('Gravity Magnitude Field', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X Position', fontsize=12)
    ax1.set_ylabel('Y Position', fontsize=12)
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label('||g|| (Gravity Strength)', fontsize=11)

    # Add celestial bodies
    for body in env.celestial_bodies:
        if body.type == 'black_hole':
            color = 'black'
            marker = 'o'
            size = 200
            label = 'Black Hole'
            # Draw event horizon
            circle = Circle((body.x, body.y), body.event_horizon,
                          fill=False, edgecolor='white', linewidth=2, linestyle='--')
            ax1.add_patch(circle)
        elif body.type == 'target_planet':
            color = 'lime'
            marker = '*'
            size = 300
            label = 'Target Planet'
        else:
            color = 'cyan'
            marker = 'o'
            size = 150
            label = 'Normal Planet'

        ax1.scatter(body.x, body.y, c=color, marker=marker, s=size,
                   edgecolors='white', linewidths=2, label=label, zorder=10)

    # Remove duplicate labels
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=10)

    # --- Plot 2: Gravity Vector Field (Quiver Plot) ---
    ax2 = axes[1]

    # Subsample for clarity (every 5th point)
    step = 5
    X, Y = np.meshgrid(np.arange(0, env.grid_size, step), np.arange(0, env.grid_size, step))

    # Get gravity vectors at sampled points
    U = gravity_field[::step, ::step, 0].T
    V = gravity_field[::step, ::step, 1].T

    # Normalize for visualization (keep direction, scale arrows)
    magnitude = np.sqrt(U**2 + V**2)
    magnitude = np.where(magnitude == 0, 1, magnitude)  # Avoid division by zero
    U_norm = U / magnitude
    V_norm = V / magnitude

    # Color by magnitude
    ax2.quiver(X, Y, U_norm, V_norm, magnitude, cmap='viridis',
              scale=30, width=0.003, alpha=0.7)
    ax2.set_title('Gravity Vector Field', fontsize=14, fontweight='bold')
    ax2.set_xlabel('X Position', fontsize=12)
    ax2.set_ylabel('Y Position', fontsize=12)
    ax2.set_xlim(0, env.grid_size)
    ax2.set_ylim(0, env.grid_size)

    # Add celestial bodies
    for body in env.celestial_bodies:
        if body.type == 'black_hole':
            color = 'black'
            marker = 'o'
            size = 200
            # Draw event horizon
            circle = Circle((body.x, body.y), body.event_horizon,
                          fill=False, edgecolor='red', linewidth=2, linestyle='--')
            ax2.add_patch(circle)
        elif body.type == 'target_planet':
            color = 'lime'
            marker = '*'
            size = 300
        else:
            color = 'orange'
            marker = 'o'
            size = 150

        ax2.scatter(body.x, body.y, c=color, marker=marker, s=size,
                   edgecolors='black', linewidths=2, zorder=10)

    ax2.grid(alpha=0.3)

    # --- Plot 3: Gravity Magnitude with Contour Lines ---
    ax3 = axes[2]

    # Log scale for better visualization
    log_magnitude = np.log10(gravity_magnitude + 1e-6)

    im3 = ax3.imshow(log_magnitude.T, origin='lower', cmap='plasma', interpolation='bilinear')
    ax3.set_title('Log10 Gravity Magnitude + Contours', fontsize=14, fontweight='bold')
    ax3.set_xlabel('X Position', fontsize=12)
    ax3.set_ylabel('Y Position', fontsize=12)

    # Add contour lines
    contour_levels = np.linspace(log_magnitude.min(), log_magnitude.max(), 15)
    contours = ax3.contour(log_magnitude.T, levels=contour_levels,
                          colors='white', alpha=0.4, linewidths=0.5, origin='lower')
    ax3.clabel(contours, inline=True, fontsize=8, fmt='%.1f')

    cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    cbar3.set_label('log10(||g||)', fontsize=11)

    # Add celestial bodies
    for body in env.celestial_bodies:
        if body.type == 'black_hole':
            color = 'white'
            marker = 'o'
            size = 200
            # Draw event horizon
            circle = Circle((body.x, body.y), body.event_horizon,
                          fill=False, edgecolor='red', linewidth=3, linestyle='--')
            ax3.add_patch(circle)
        elif body.type == 'target_planet':
            color = 'lime'
            marker = '*'
            size = 300
        else:
            color = 'yellow'
            marker = 'o'
            size = 150

        ax3.scatter(body.x, body.y, c=color, marker=marker, s=size,
                   edgecolors='black', linewidths=2, zorder=10)

    plt.suptitle('Gravitational Dynamics Environment - Gravity Field Analysis',
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")

    plt.show()


def visualize_gravity_cross_sections(env: GravitationalDynamicsEnv, save_path: str = None):
    """
    Visualize gravity field cross-sections along x and y axes.
    """
    gravity_field = env.get_gravity_field()

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # --- Cross-section along x-axis at y=50 ---
    ax1 = axes[0, 0]
    y_slice = 50
    gx_slice = gravity_field[:, y_slice, 0]
    gy_slice = gravity_field[:, y_slice, 1]
    g_mag_slice = np.sqrt(gx_slice**2 + gy_slice**2)

    x_coords = np.arange(env.grid_size)
    ax1.plot(x_coords, gx_slice, label='gx', linewidth=2)
    ax1.plot(x_coords, gy_slice, label='gy', linewidth=2)
    ax1.plot(x_coords, g_mag_slice, label='||g||', linewidth=2, linestyle='--', color='black')
    ax1.set_xlabel('X Position', fontsize=12)
    ax1.set_ylabel('Gravity Components', fontsize=12)
    ax1.set_title(f'Gravity Cross-Section along X-axis (y={y_slice})', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)

    # Mark celestial bodies on this line
    for body in env.celestial_bodies:
        if abs(body.y - y_slice) < 5:  # Within 5 units
            ax1.axvline(body.x, color='red', linestyle=':', alpha=0.5, linewidth=1.5)
            ax1.text(body.x, ax1.get_ylim()[1] * 0.9, body.type[:3],
                    rotation=90, fontsize=8, ha='right')

    # --- Cross-section along y-axis at x=50 ---
    ax2 = axes[0, 1]
    x_slice = 50
    gx_slice = gravity_field[x_slice, :, 0]
    gy_slice = gravity_field[x_slice, :, 1]
    g_mag_slice = np.sqrt(gx_slice**2 + gy_slice**2)

    y_coords = np.arange(env.grid_size)
    ax2.plot(y_coords, gx_slice, label='gx', linewidth=2)
    ax2.plot(y_coords, gy_slice, label='gy', linewidth=2)
    ax2.plot(y_coords, g_mag_slice, label='||g||', linewidth=2, linestyle='--', color='black')
    ax2.set_xlabel('Y Position', fontsize=12)
    ax2.set_ylabel('Gravity Components', fontsize=12)
    ax2.set_title(f'Gravity Cross-Section along Y-axis (x={x_slice})', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)

    # Mark celestial bodies on this line
    for body in env.celestial_bodies:
        if abs(body.x - x_slice) < 5:
            ax2.axvline(body.y, color='red', linestyle=':', alpha=0.5, linewidth=1.5)
            ax2.text(body.y, ax2.get_ylim()[1] * 0.9, body.type[:3],
                    rotation=90, fontsize=8, ha='right')

    # --- Gravity magnitude distribution histogram ---
    ax3 = axes[1, 0]
    gravity_magnitude = np.sqrt(gravity_field[:, :, 0]**2 + gravity_field[:, :, 1]**2)
    ax3.hist(gravity_magnitude.flatten(), bins=100, color='steelblue', edgecolor='black', alpha=0.7)
    ax3.set_xlabel('Gravity Magnitude ||g||', fontsize=12)
    ax3.set_ylabel('Frequency', fontsize=12)
    ax3.set_title('Distribution of Gravity Magnitude', fontsize=13, fontweight='bold')
    ax3.set_yscale('log')
    ax3.grid(alpha=0.3)

    # --- Celestial body information table ---
    ax4 = axes[1, 1]
    ax4.axis('off')

    table_data = []
    table_data.append(['Type', 'Position', 'Mass', 'Reward', 'Event Horizon'])

    for body in env.celestial_bodies:
        row = [
            body.type,
            f"({body.x:.1f}, {body.y:.1f})",
            f"{body.mass:.1e}",
            f"{body.reward:.1f}",
            f"{body.event_horizon}" if body.event_horizon else "N/A"
        ]
        table_data.append(row)

    table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.25, 0.2, 0.2, 0.15, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header row
    for i in range(5):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Color code rows by type
    for i, body in enumerate(env.celestial_bodies, start=1):
        if body.type == 'black_hole':
            color = '#FFE5E5'
        elif body.type == 'target_planet':
            color = '#E5FFE5'
        else:
            color = '#E5F2FF'

        for j in range(5):
            table[(i, j)].set_facecolor(color)

    ax4.set_title('Celestial Bodies Information', fontsize=13, fontweight='bold', pad=20)

    plt.suptitle('Gravitational Field Analysis - Detailed View',
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Cross-section visualization saved to: {save_path}")

    plt.show()


if __name__ == "__main__":
    print("Creating gravitational dynamics environment...")
    env = GravitationalDynamicsEnv()

    print("\n=== Environment Statistics ===")
    gravity_field = env.get_gravity_field()
    gravity_magnitude = np.sqrt(gravity_field[:, :, 0]**2 + gravity_field[:, :, 1]**2)

    print(f"Grid size: {env.grid_size}x{env.grid_size}")
    print(f"Gravitational constant G: {env.G}")
    print(f"Number of celestial bodies: {len(env.celestial_bodies)}")
    print(f"\nGravity field statistics:")
    print(f"  Min magnitude: {gravity_magnitude.min():.6f}")
    print(f"  Max magnitude: {gravity_magnitude.max():.6f}")
    print(f"  Mean magnitude: {gravity_magnitude.mean():.6f}")
    print(f"  Median magnitude: {np.median(gravity_magnitude):.6f}")

    print("\n=== Celestial Bodies ===")
    for body in env.celestial_bodies:
        print(f"{body.type}:")
        print(f"  Position: ({body.x}, {body.y})")
        print(f"  Mass: {body.mass:.2e}")
        print(f"  Reward: {body.reward}")
        if body.event_horizon:
            print(f"  Event Horizon: {body.event_horizon} units")
        print()

    print("\nGenerating visualizations...")
    print("\n[1] Main gravity field visualization...")
    visualize_gravity_field(env, save_path='gravitational/gravity_field.png')

    print("\n[2] Cross-section analysis...")
    visualize_gravity_cross_sections(env, save_path='gravitational/gravity_cross_sections.png')

    print("\nVisualization complete!")
