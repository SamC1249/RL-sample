"""
Environment Visualization Tool
Visualizes the 1000x1000 grid with celestial objects and gravity fields
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.colors as mcolors
from astrophysics_env import AstrophysicsEnv
import argparse


def compute_gravity_field(env, resolution=100):
    """
    Compute the gravitational force magnitude at each point in the grid

    Args:
        env: AstrophysicsEnv instance
        resolution: Grid resolution for computation (higher = slower but more detailed)

    Returns:
        x_grid, y_grid, gravity_magnitude
    """
    print(f"Computing gravity field at {resolution}x{resolution} resolution...")

    # Create meshgrid
    x = np.linspace(0, env.grid_size, resolution)
    y = np.linspace(0, env.grid_size, resolution)
    X, Y = np.meshgrid(x, y)

    # Initialize gravity field
    gravity_magnitude = np.zeros_like(X)

    # Compute gravity from each sun
    for sun in env.suns:
        dx = X - sun.x
        dy = Y - sun.y
        dist = np.sqrt(dx**2 + dy**2)

        # Avoid division by zero
        dist = np.maximum(dist, 1.0)

        # Only apply gravity within influence radius
        influence_mask = dist < sun.influence_radius
        force = np.zeros_like(dist)
        force[influence_mask] = sun.gravity_strength / (dist[influence_mask] ** 2) * 100

        gravity_magnitude += force

    # Compute gravity from each black hole
    for bh in env.black_holes:
        dx = X - bh.x
        dy = Y - bh.y
        dist = np.sqrt(dx**2 + dy**2)

        # Avoid division by zero
        dist = np.maximum(dist, 1.0)

        # Only apply gravity within influence radius
        influence_mask = dist < bh.influence_radius
        force = np.zeros_like(dist)
        force[influence_mask] = bh.gravity_strength / (dist[influence_mask] ** 2) * 100

        gravity_magnitude += force

    print("Gravity field computation complete!")
    return X, Y, gravity_magnitude


def visualize_objects_only(env, ax, show_influence=True):
    """
    Visualize only the celestial objects

    Args:
        env: AstrophysicsEnv instance
        ax: Matplotlib axis
        show_influence: Whether to show influence radius circles
    """
    ax.set_xlim(0, env.grid_size)
    ax.set_ylim(0, env.grid_size)
    ax.set_aspect('equal')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('Celestial Objects Map')
    ax.grid(True, alpha=0.2)

    # Draw Earth
    earth = Circle(env.earth_pos, 10, color='#1E90FF', label='Earth (Start)', zorder=5)
    ax.add_patch(earth)
    ax.plot(env.earth_pos[0], env.earth_pos[1], 'b*', markersize=15, zorder=6)

    # Draw black holes
    for i, bh in enumerate(env.black_holes):
        if show_influence:
            # Influence radius (very faint)
            influence = Circle((bh.x, bh.y), bh.influence_radius,
                             color='purple', alpha=0.05, linestyle='--', fill=False, linewidth=1)
            ax.add_patch(influence)

        # Event horizon (more visible)
        event_horizon = Circle((bh.x, bh.y), bh.event_horizon,
                              color='purple', alpha=0.3, label='Black Hole Event Horizon' if i == 0 else '')
        ax.add_patch(event_horizon)

        # Center (black)
        center = Circle((bh.x, bh.y), bh.radius,
                       color='black', label='Black Hole Center' if i == 0 else '', zorder=4)
        ax.add_patch(center)

        # Add label
        ax.text(bh.x, bh.y - bh.event_horizon - 15, f'BH{i+1}',
               ha='center', fontsize=8, color='purple', fontweight='bold')

    # Draw suns
    for i, sun in enumerate(env.suns):
        if show_influence:
            # Influence radius (very faint)
            influence = Circle((sun.x, sun.y), sun.influence_radius,
                             color='yellow', alpha=0.05, linestyle='--', fill=False, linewidth=1)
            ax.add_patch(influence)

        # Sun center
        sun_circle = Circle((sun.x, sun.y), sun.radius,
                           color='#FFD700', edgecolor='orange', linewidth=2,
                           label='Sun' if i == 0 else '', zorder=3)
        ax.add_patch(sun_circle)

        # Add label
        ax.text(sun.x, sun.y - sun.influence_radius - 15, f'S{i+1}',
               ha='center', fontsize=8, color='orange', fontweight='bold')

    # Draw landable planets (green)
    for i, planet in enumerate(env.landable_planets):
        planet_circle = Circle((planet.x, planet.y), planet.radius * 1.5,
                              color='#32CD32', edgecolor='darkgreen', linewidth=1.5,
                              label='Landable Planet' if i == 0 else '', zorder=2)
        ax.add_patch(planet_circle)

    # Draw non-landable planets (gray)
    for i, planet in enumerate(env.non_landable_planets):
        planet_circle = Circle((planet.x, planet.y), planet.radius,
                              color='gray', alpha=0.6,
                              label='Non-landable Planet' if i == 0 else '', zorder=1)
        ax.add_patch(planet_circle)

    # Draw asteroids (brown)
    for i, asteroid in enumerate(env.asteroids):
        asteroid_circle = Circle((asteroid.x, asteroid.y), asteroid.radius,
                                color='#8B4513', alpha=0.5,
                                label='Asteroid' if i == 0 else '', zorder=1)
        ax.add_patch(asteroid_circle)

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    # Remove duplicates
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=8)


def visualize_gravity_field(env, ax, resolution=100):
    """
    Visualize the gravitational field as a heatmap

    Args:
        env: AstrophysicsEnv instance
        ax: Matplotlib axis
        resolution: Grid resolution for gravity computation
    """
    X, Y, gravity = compute_gravity_field(env, resolution)

    # Plot gravity field as heatmap
    im = ax.contourf(X, Y, gravity, levels=20, cmap='YlOrRd', alpha=0.7)
    plt.colorbar(im, ax=ax, label='Gravitational Force Magnitude')

    # Add contour lines
    contours = ax.contour(X, Y, gravity, levels=10, colors='black', alpha=0.3, linewidths=0.5)
    ax.clabel(contours, inline=True, fontsize=6)

    ax.set_xlim(0, env.grid_size)
    ax.set_ylim(0, env.grid_size)
    ax.set_aspect('equal')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('Gravitational Field Strength')
    ax.grid(True, alpha=0.2)


def visualize_combined(env, ax, resolution=100, show_gravity=True):
    """
    Combine gravity field and objects in one view

    Args:
        env: AstrophysicsEnv instance
        ax: Matplotlib axis
        resolution: Grid resolution for gravity computation
        show_gravity: Whether to show gravity field
    """
    if show_gravity:
        X, Y, gravity = compute_gravity_field(env, resolution)

        # Plot gravity field as semi-transparent heatmap
        im = ax.contourf(X, Y, gravity, levels=15, cmap='YlOrRd', alpha=0.4)
        plt.colorbar(im, ax=ax, label='Gravity Force')

    # Draw all objects on top
    ax.set_xlim(0, env.grid_size)
    ax.set_ylim(0, env.grid_size)
    ax.set_aspect('equal')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('Combined View: Objects + Gravity Field')
    ax.grid(True, alpha=0.2, color='white', linewidth=0.5)

    # Earth
    earth = Circle(env.earth_pos, 10, color='#1E90FF', edgecolor='white', linewidth=2, zorder=10)
    ax.add_patch(earth)
    ax.plot(env.earth_pos[0], env.earth_pos[1], 'w*', markersize=20, zorder=11,
            markeredgecolor='black', markeredgewidth=1)
    ax.text(env.earth_pos[0], env.earth_pos[1] - 25, 'EARTH',
           ha='center', fontsize=10, color='white', fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='blue', alpha=0.7))

    # Black holes
    for i, bh in enumerate(env.black_holes):
        # Event horizon
        event_horizon = Circle((bh.x, bh.y), bh.event_horizon,
                              color='purple', alpha=0.5, edgecolor='white', linewidth=2)
        ax.add_patch(event_horizon)

        # Center
        center = Circle((bh.x, bh.y), bh.radius,
                       color='black', edgecolor='white', linewidth=2, zorder=9)
        ax.add_patch(center)

        ax.text(bh.x, bh.y, f'BH{i+1}', ha='center', va='center',
               fontsize=8, color='white', fontweight='bold')

    # Suns
    for i, sun in enumerate(env.suns):
        sun_circle = Circle((sun.x, sun.y), sun.radius * 1.5,
                           color='#FFD700', edgecolor='orange', linewidth=2, zorder=8)
        ax.add_patch(sun_circle)

        ax.text(sun.x, sun.y, f'S{i+1}', ha='center', va='center',
               fontsize=7, color='black', fontweight='bold')

    # Landable planets
    for planet in env.landable_planets:
        planet_circle = Circle((planet.x, planet.y), planet.radius * 2,
                              color='#00FF00', edgecolor='darkgreen', linewidth=2,
                              alpha=0.8, zorder=7)
        ax.add_patch(planet_circle)

    # Non-landable planets (smaller, more transparent)
    for planet in env.non_landable_planets:
        planet_circle = Circle((planet.x, planet.y), planet.radius,
                              color='gray', alpha=0.4, zorder=6)
        ax.add_patch(planet_circle)

    # Asteroids (very small)
    for asteroid in env.asteroids:
        asteroid_circle = Circle((asteroid.x, asteroid.y), asteroid.radius,
                                color='brown', alpha=0.4, zorder=5)
        ax.add_patch(asteroid_circle)


def visualize_environment_stats(env, ax):
    """
    Display environment statistics

    Args:
        env: AstrophysicsEnv instance
        ax: Matplotlib axis
    """
    ax.axis('off')
    ax.set_title('Environment Statistics', fontsize=14, fontweight='bold')

    stats_text = f"""
ENVIRONMENT CONFIGURATION
{'='*40}

Grid Size: {env.grid_size}x{env.grid_size}
Max Fuel: {env.max_fuel}
Max Steps: {env.max_steps}
Seed: {env.seed_value}
Mode: {'Static' if env.static_environment else 'Dynamic'}

CELESTIAL OBJECTS
{'='*40}

Earth: 1 (Start position)
  • Position: ({env.earth_pos[0]:.0f}, {env.earth_pos[1]:.0f})

Black Holes: {len(env.black_holes)}
  • Gravity Strength: 2.0
  • Influence Radius: 100 units
  • Event Horizon: 10 units
  • Collision Penalty: -100

Suns: {len(env.suns)}
  • Gravity Strength: 0.5
  • Influence Radius: 50 units
  • Collision Penalty: -100

Landable Planets: {len(env.landable_planets)}
  • Landing Reward: 0 (Success!)
  • Terminal State: Yes

Non-landable Planets: {len(env.non_landable_planets)}
  • Obstacles only

Asteroids: {len(env.asteroids)}
  • Collision Penalty: -2

REWARD STRUCTURE
{'='*40}

• Base action cost: -1
• Asteroid hit: -2
• Black hole influence: -2/step
• Sun/BH collision: -100
• Out of fuel: -2/step
• Successful landing: 0 (terminal)
"""

    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
           fontsize=9, verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


def main():
    parser = argparse.ArgumentParser(description='Visualize Astrophysics Environment')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for environment generation (default: 42)')
    parser.add_argument('--resolution', type=int, default=100,
                       help='Grid resolution for gravity field (default: 100, higher=slower)')
    parser.add_argument('--no-gravity', action='store_true',
                       help='Do not compute/show gravity field (faster)')
    parser.add_argument('--dynamic', action='store_true',
                       help='Use dynamic environment mode')
    parser.add_argument('--save', type=str, default=None,
                       help='Save visualization to file (e.g., environment.png)')

    args = parser.parse_args()

    print("="*60)
    print("ASTROPHYSICS ENVIRONMENT VISUALIZATION")
    print("="*60)

    # Create environment
    print(f"\nCreating environment with seed {args.seed}...")
    env = AstrophysicsEnv(
        seed=args.seed,
        static_environment=not args.dynamic
    )
    env.reset()

    print(f"Environment created!")
    print(f"  - Black Holes: {len(env.black_holes)}")
    print(f"  - Suns: {len(env.suns)}")
    print(f"  - Landable Planets: {len(env.landable_planets)}")
    print(f"  - Non-landable Planets: {len(env.non_landable_planets)}")
    print(f"  - Asteroids: {len(env.asteroids)}")

    # Create figure with multiple subplots
    if args.no_gravity:
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # Objects only
        visualize_objects_only(env, axes[0], show_influence=True)

        # Stats
        visualize_environment_stats(env, axes[1])
    else:
        fig = plt.figure(figsize=(20, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

        # Objects only
        ax1 = fig.add_subplot(gs[0, 0])
        visualize_objects_only(env, ax1, show_influence=True)

        # Gravity field only
        ax2 = fig.add_subplot(gs[0, 1])
        visualize_gravity_field(env, ax2, resolution=args.resolution)

        # Combined view
        ax3 = fig.add_subplot(gs[0, 2])
        visualize_combined(env, ax3, resolution=args.resolution, show_gravity=True)

        # Stats (spans bottom row)
        ax4 = fig.add_subplot(gs[1, :])
        visualize_environment_stats(env, ax4)

    plt.suptitle(f'Astrophysics Environment Visualization (Seed: {args.seed})',
                fontsize=16, fontweight='bold')

    if args.save:
        print(f"\nSaving visualization to {args.save}...")
        plt.savefig(args.save, dpi=150, bbox_inches='tight')
        print(f"Saved!")

    print("\nDisplaying visualization...")
    print("Close the window to exit.")
    plt.show()

    env.close()


if __name__ == "__main__":
    main()
