## Figure 1: FEM models of stomata using realistic geometry open without anisotropic stiffening or polar pinning
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import colorsys
import matplotlib.colors as mcolors

def run_plot_fig1(sample_meshes= ["1_2","2_3","3_2"], confocal_df = None, mesh_dcr = "Sample", colours=None):
    all_meshes = confocal_df["Mesh ID"].unique()
    number_of_samples = 3
    mesh_names = [mesh_dcr + " " + str(s) for s in np.arange(1, number_of_samples + 1)]
    symbols = ["*","^","s"]
    sample_lightness = [1.0, 1.0, 1.0]

    base_colour = mcolors.to_rgb(colours['empirical'])

    def adjust_lightness(rgb, factor):
        h, l, s = colorsys.rgb_to_hls(*rgb)
        l = min(1.0, max(0.0, l * factor))
        return colorsys.hls_to_rgb(h, l, s)

    ## Plot pressure vs pore area for each of the sample meshes
    for mesh, symbol, mesh_name, factor in zip(sample_meshes, symbols, mesh_names, sample_lightness):
        data = confocal_df[confocal_df["Mesh ID"] == mesh]
        plt.scatter(
            data["Pressure"],
            data["Pore Area"],
            marker=symbol,
            label=mesh_name,
            #facecolors=adjust_lightness(base_colour, factor),
            facecolors = base_colour,
            s=120,
            edgecolors="k",
            linewidths=1.2,
        )
    plt.xlabel("Pressure increase (MPa)")
    plt.ylabel(r'Pore Area ($\mu\mathrm{m}^2$)')
    plt.xlim(0,2.1)
    ax = plt.gca()
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0., fontsize=10)
    plt.tight_layout()
    plt.savefig("../Figures/Fig1/confocal_pressure_pore_area.png", dpi=300, bbox_inches='tight')
    plt.show()

    ## Plot starting pore area vs end pore area / start pore area. Highlight the three selected meshes
    for mesh in all_meshes:
        data = confocal_df[confocal_df["Mesh ID"]==mesh]
        start_pore = data[data["Pressure"]==0.0]["Pore Area"].values
        end_pore = data[data["Pressure"]==2.0]["Pore Area"].values
        marker_colour = base_colour
        edge_colour = "none"
        if mesh == sample_meshes[0]: 
            marker = symbols[0]
            size = 17
            #marker_colour = adjust_lightness(base_colour, sample_lightness[0])
            marker_colour = base_colour
            edge_colour = "k"
        elif mesh == sample_meshes[1]: 
            marker = symbols[1]
            size = 17
            #marker_colour = adjust_lightness(base_colour, sample_lightness[1])
            marker_colour = base_colour
            edge_colour = "k"
        elif mesh == sample_meshes[2]: 
            marker = symbols[2]
            size = 12
            #marker_colour = adjust_lightness(base_colour, sample_lightness[2])
            marker_colour = base_colour
            edge_colour = "k"
        else: 
            marker = "o"
            size = 10
            #marker_colour = adjust_lightness(base_colour, 0.45)
            marker_colour = base_colour
            edge_colour = "k"

        plt.plot(
            start_pore,
            end_pore/start_pore,
            marker=marker,
            linestyle="None",
            markerfacecolor=marker_colour,
            markeredgecolor=edge_colour,
            markeredgewidth=1.2,
            markersize=size,
        )
    legend_elements = []
    for name, symbol, factor in zip(mesh_names, symbols, sample_lightness):
        legend_elements.append(
            Line2D(
                [0],
                [0],
                marker=symbol,
                linestyle='None',
                markerfacecolor=base_colour,
                markeredgecolor='k',
                markeredgewidth=1.2,
                markersize=10,
                label=name,
            )
        )

    legend_elements.append(
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=base_colour,
            markeredgecolor="k",
            markersize=8,
            label="Other meshes",
        )
    )

    ax = plt.gca()
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0., fontsize=10)
    plt.xlabel(r'Starting pore area ($\mu\mathrm{m}^2$)')
    plt.ylabel("End pore area / Start pore area")
    plt.tight_layout()
    plt.savefig("../Figures/Fig1/confocal_change_pore_area.png", dpi=300, bbox_inches='tight')
