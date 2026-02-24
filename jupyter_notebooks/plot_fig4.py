import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def check_data(results_matrix_iso, results_matrix_aniso, selected_meshes):
    ## First, let's check for any problematic values in the results matrices
    print("Checking results_matrix_iso for zeros or NaNs:")
    print(f"  Zeros: {np.sum(results_matrix_iso == 0)}")
    print(f"  NaNs: {np.sum(np.isnan(results_matrix_iso))}")
    print(f"  Shape: {results_matrix_iso.shape}")
    print(f"  Min value: {np.nanmin(results_matrix_iso)}")
    print(f"  Max value: {np.nanmax(results_matrix_iso)}")

    print("\nChecking results_matrix_aniso for zeros or NaNs:")
    print(f"  Zeros: {np.sum(results_matrix_aniso == 0)}")
    print(f"  NaNs: {np.sum(np.isnan(results_matrix_aniso))}")
    print(f"  Shape: {results_matrix_aniso.shape}")
    print(f"  Min value: {np.nanmin(results_matrix_aniso)}")
    print(f"  Max value: {np.nanmax(results_matrix_aniso)}")

    # Check if any rows are all zeros (indicating a completely failed mesh)
    iso_failed_meshes = np.where(np.all(results_matrix_iso == 0, axis=1))[0]
    aniso_failed_meshes = np.where(np.all(results_matrix_aniso == 0, axis=1))[0]

    if len(iso_failed_meshes) > 0:
        print(f"\nISO: Meshes with all zeros: {[selected_meshes[i] for i in iso_failed_meshes]}")
    if len(aniso_failed_meshes) > 0:
        print(f"ANISO: Meshes with all zeros: {[selected_meshes[i] for i in aniso_failed_meshes]}")

def calculate_means(confocal_df, confocal_df_aniso, selected_meshes, results_matrix_iso, results_matrix_aniso):

    for mesh in selected_meshes:
        mesh_mask = confocal_df["Mesh ID"].str.contains(mesh)
        mesh_data = confocal_df[mesh_mask]
        start_area = mesh_data[mesh_data["Pressure"] == 0.0]["Pore Area"].values
        if len(start_area) == 0:
            continue
        start_area = start_area[0]
        confocal_df.loc[mesh_mask, "Pore Area Change"] = mesh_data["Pore Area"] / start_area

        mesh_mask_aniso = confocal_df_aniso["Mesh ID"].str.contains(mesh)
        mesh_data_aniso = confocal_df_aniso[mesh_mask_aniso]
        start_area_aniso = mesh_data_aniso[mesh_data_aniso["Pressure"] == 0.0]["Pore Area"].values
        if len(start_area_aniso) == 0:
            continue
        start_area_aniso = start_area_aniso[0]
        confocal_df_aniso.loc[mesh_mask_aniso, "Pore Area Change"] = mesh_data_aniso["Pore Area"] / start_area_aniso

    pressures = np.round(np.arange(0, 2.1, 0.1), 1)

    # Collect data into arrays for each pressure point
    mean_values_iso = []
    sem_values_iso = []
    sd_values_iso = []
    mean_values_aniso = []
    sem_values_aniso = []
    sd_values_aniso = []

    for p in pressures:
        iso_data = confocal_df[confocal_df["Pressure"] == p]["Pore Area Change"].dropna().values
        aniso_data = confocal_df_aniso[confocal_df_aniso["Pressure"] == p]["Pore Area Change"].dropna().values
        
        if len(iso_data) > 0:
            mean_values_iso.append(np.mean(iso_data))
            sem_values_iso.append(stats.sem(iso_data))
            sd_values_iso.append(np.std(iso_data))
        else:
            mean_values_iso.append(np.nan)
            sem_values_iso.append(np.nan)
            sd_values_iso.append(np.nan)
        
        if len(aniso_data) > 0:
            mean_values_aniso.append(np.mean(aniso_data))
            sem_values_aniso.append(stats.sem(aniso_data))
            sd_values_aniso.append(np.std(aniso_data))
        else:
            mean_values_aniso.append(np.nan)
            sem_values_aniso.append(np.nan)
            sd_values_aniso.append(np.nan)

    mean_values_iso = np.array(mean_values_iso)
    sem_values_iso = np.array(sem_values_iso)
    sd_values_iso = np.array(sd_values_iso)
    mean_values_aniso = np.array(mean_values_aniso)
    sem_values_aniso = np.array(sem_values_aniso)
    sd_values_aniso = np.array(sd_values_aniso)

    # Mask out zeros and compute statistics
    results_matrix_iso_masked = np.where(results_matrix_iso == 0, np.nan, results_matrix_iso)
    results_matrix_aniso_masked = np.where(results_matrix_aniso == 0, np.nan, results_matrix_aniso)

    # Normalize by starting length for each mesh
    results_matrix_iso_norm = results_matrix_iso_masked / results_matrix_iso_masked[:, 0:1]
    results_matrix_aniso_norm = results_matrix_aniso_masked / results_matrix_aniso_masked[:, 0:1]

    # Compute mean and standard error
    mean_iso = np.nanmean(results_matrix_iso_norm, axis=0)
    sem_iso = stats.sem(results_matrix_iso_norm, axis=0, nan_policy='omit')
    sd_iso = np.nanstd(results_matrix_iso_norm, axis=0)

    mean_aniso = np.nanmean(results_matrix_aniso_norm, axis=0)
    sem_aniso = stats.sem(results_matrix_aniso_norm, axis=0, nan_policy='omit')
    sd_aniso = np.nanstd(results_matrix_aniso_norm, axis=0)

    return(mean_values_iso, sem_values_iso, mean_values_aniso, sem_values_aniso, mean_iso, sem_iso, sd_iso, mean_aniso, sem_aniso, sd_aniso, confocal_df, confocal_df_aniso)

def run_plot_fig4(results_matrix_iso, results_matrix_aniso, selected_meshes, confocal_df, confocal_df_aniso, pressures, colours):
    # First, check the data for any issues
    check_data(results_matrix_iso, results_matrix_aniso, selected_meshes)
    mean_values_iso, sem_values_iso, mean_values_aniso, sem_values_aniso, mean_iso, sem_iso, sd_iso, mean_aniso, sem_aniso, sd_aniso, confocal_df, confocal_df_aniso = calculate_means(confocal_df, confocal_df_aniso, selected_meshes, results_matrix_iso, results_matrix_aniso)

    ## Combined plot: Guard Cell Length and Pore Area Change with SEM
    # Using dual y-axes for better visualization

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # ============ LEFT PANEL: Isotropic ============
    # Create primary axis for Pore Area
    ax1_pore = ax1
    ax1_pore.plot(pressures, mean_values_iso, '-', color=colours['empirical'], linewidth=3, 
                label='Pore Area', zorder=3, marker='o', markersize=6)
    ax1_pore.fill_between(pressures, mean_values_iso - sem_values_iso, mean_values_iso + sem_values_iso, 
                        color=colours['empirical'], alpha=0.3, zorder=1)
    ax1_pore.set_xlabel("Pressure (MPa)", fontsize=16)
    ax1_pore.set_ylabel("Pore Area Change", fontsize=16, color=colours['empirical'])
    ax1_pore.tick_params(axis='y', labelcolor=colours['empirical'])

    # Create secondary axis for GC Length
    ax1_gc = ax1.twinx()
    ax1_gc.plot(pressures, mean_iso, '--', color='darkblue', linewidth=3, 
                label='GC Length', zorder=3, marker='s', markersize=6)
    ax1_gc.fill_between(pressures, mean_iso - sem_iso, mean_iso + sem_iso, 
                        color='darkblue', alpha=0.3, zorder=1)
    ax1_gc.set_ylabel("GC Length Change", fontsize=16, color='darkblue')
    ax1_gc.tick_params(axis='y', labelcolor='darkblue')

    # Reference lines
    ax1_pore.axhline(y=1, color='k', linestyle=':', linewidth=1.5, alpha=0.5, zorder=2)
    ax1_gc.axhline(y=1, color='k', linestyle=':', linewidth=1.5, alpha=0.5, zorder=2)

    ax1_pore.set_title("Isotropic Stiffening", fontsize=16, fontweight='bold')
    ax1_pore.grid(True, alpha=0.3)

    # Combined legend
    lines1, labels1 = ax1_pore.get_legend_handles_labels()
    lines2, labels2 = ax1_gc.get_legend_handles_labels()
    ax1_pore.legend(lines1 + lines2, labels1 + labels2, fontsize=12, loc='upper left')

    # ============ RIGHT PANEL: Anisotropic ============
    # Create primary axis for Pore Area
    ax2_pore = ax2
    ax2_pore.plot(pressures, mean_values_aniso, '-', color=colours['anisotropic'], linewidth=3, 
                label='Pore Area', zorder=3, marker='o', markersize=6)
    ax2_pore.fill_between(pressures, mean_values_aniso - sem_values_aniso, mean_values_aniso + sem_values_aniso, 
                        color=colours['anisotropic'], alpha=0.3, zorder=1)
    ax2_pore.set_xlabel("Pressure (MPa)", fontsize=16)
    ax2_pore.set_ylabel("Pore Area Change", fontsize=16, color=colours['anisotropic'])
    ax2_pore.tick_params(axis='y', labelcolor=colours['anisotropic'])

    # Create secondary axis for GC Length
    ax2_gc = ax2.twinx()
    ax2_gc.plot(pressures, mean_aniso, '--', color='darkgreen', linewidth=3, 
                label='GC Length', zorder=3, marker='s', markersize=6)
    ax2_gc.fill_between(pressures, mean_aniso - sem_aniso, mean_aniso + sem_aniso, 
                        color='darkgreen', alpha=0.3, zorder=1)
    ax2_gc.set_ylabel("GC Length Change", fontsize=16, color='darkgreen')
    ax2_gc.tick_params(axis='y', labelcolor='darkgreen')

    # Reference lines
    ax2_pore.axhline(y=1, color='k', linestyle=':', linewidth=1.5, alpha=0.5, zorder=2)
    ax2_gc.axhline(y=1, color='k', linestyle=':', linewidth=1.5, alpha=0.5, zorder=2)

    ax2_pore.set_title("Anisotropic Stiffening", fontsize=16, fontweight='bold')
    ax2_pore.grid(True, alpha=0.3)

    # Combined legend
    lines1, labels1 = ax2_pore.get_legend_handles_labels()
    lines2, labels2 = ax2_gc.get_legend_handles_labels()
    ax2_pore.legend(lines1 + lines2, labels1 + labels2, fontsize=16, loc='upper left')

    plt.tight_layout()
    plt.savefig("combined_gc_length_pore_area_dual_axes.png", dpi=300, bbox_inches='tight')
    plt.show()

    # ============ ALTERNATIVE: Single plot with dual axes ============
    fig, ax_pore = plt.subplots(figsize=(10, 7))

    # Pore Area on left y-axis
    ax_pore.plot(pressures, mean_values_iso, '-', color=colours['empirical'], linewidth=3, 
                label='Pore Area (Isotropic)', zorder=4, marker='o', markersize=7)
    ax_pore.fill_between(pressures, mean_values_iso - sem_values_iso, mean_values_iso + sem_values_iso, 
                        color=colours['empirical'], alpha=0.25, zorder=1)

    ax_pore.plot(pressures, mean_values_aniso, '-', color=colours['anisotropic'], linewidth=3, 
                label='Pore Area (Anisotropic)', zorder=4, marker='o', markersize=7)
    ax_pore.fill_between(pressures, mean_values_aniso - sem_values_aniso, mean_values_aniso + sem_values_aniso, 
                        color=colours['anisotropic'], alpha=0.25, zorder=1)

    ax_pore.set_xlabel("Pressure (MPa)", fontsize=18)
    ax_pore.set_ylabel("Pore Area Change", fontsize=18)
    ax_pore.tick_params(axis='y', labelsize=16)
    ax_pore.tick_params(axis='x', labelsize=16)
    ax_pore.axhline(y=1, color='gray', linestyle=':', linewidth=1.5, alpha=0.5, zorder=2)

    # GC Length on right y-axis
    ax_gc = ax_pore.twinx()

    ax_gc.plot(pressures, mean_iso, '--', color='darkblue', linewidth=3, 
            label='GC Length (Isotropic)', zorder=3, marker='s', markersize=7)
    ax_gc.fill_between(pressures, mean_iso - sem_iso, mean_iso + sem_iso, 
                        color='darkblue', alpha=0.25, zorder=1)

    ax_gc.plot(pressures, mean_aniso, '--', color='darkgreen', linewidth=3, 
            label='GC Length (Anisotropic)', zorder=3, marker='s', markersize=7)
    ax_gc.fill_between(pressures, mean_aniso - sem_aniso, mean_aniso + sem_aniso, 
                        color='darkgreen', alpha=0.25, zorder=1)

    ax_gc.set_ylabel("GC Length Change", fontsize=18)
    ax_gc.tick_params(axis='y', labelsize=16)
    ax_gc.axhline(y=1, color='gray', linestyle=':', linewidth=1.5, alpha=0.5, zorder=2)

    # Combined legend
    lines1, labels1 = ax_pore.get_legend_handles_labels()
    lines2, labels2 = ax_gc.get_legend_handles_labels()
    ax_pore.legend(lines1 + lines2, labels1 + labels2, fontsize=16, loc='upper left', framealpha=0.95)

    ax_pore.grid(True, alpha=0.3, zorder=0)

    plt.tight_layout()
    plt.savefig("../Figures/Fig4/combined_all_metrics_dual_axes.png", dpi=300, bbox_inches='tight')
    plt.show()

    print("\n=== Summary Statistics ===")
    print(f"Final GC Length Change (Isotropic): {mean_iso[-1]:.3f} ± {sem_iso[-1]:.3f}")
    print(f"Final GC Length Change (Anisotropic): {mean_aniso[-1]:.3f} ± {sem_aniso[-1]:.3f}")
    print(f"Final Pore Area Change (Isotropic): {mean_values_iso[-1]:.3f} ± {sem_values_iso[-1]:.3f}")
    print(f"Final Pore Area Change (Anisotropic): {mean_values_aniso[-1]:.3f} ± {sem_values_aniso[-1]:.3f}")

    print("\n=== Ratio of Changes ===")
    print(f"Pore Area / GC Length (Isotropic): {(mean_values_iso[-1]-1) / (mean_iso[-1]-1):.2f}x")
    print(f"Pore Area / GC Length (Anisotropic): {(mean_values_aniso[-1]-1) / (mean_aniso[-1]-1):.2f}x")