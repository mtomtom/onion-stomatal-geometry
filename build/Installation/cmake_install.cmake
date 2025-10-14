# Install script for directory: /Users/tomkinsm/stomata-air-mattress/cgal/Installation

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Debug")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set path to fallback-tool for dependency-resolution.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "CGAL_Qt6" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CGAL/" TYPE DIRECTORY FILES
    "/Users/tomkinsm/stomata-air-mattress/cgal/Arrangement_on_surface_2/include/CGAL/Qt"
    "/Users/tomkinsm/stomata-air-mattress/cgal/Basic_viewer/include/CGAL/Qt"
    "/Users/tomkinsm/stomata-air-mattress/cgal/GraphicsView/include/CGAL/Qt"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "CGAL_Qt6" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/CGAL/demo/resources" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/GraphicsView/demo/resources/")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "CGAL_Qt6" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/CGAL/demo/icons" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/GraphicsView/demo/icons/")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/doc/CGAL" TYPE FILE FILES
    "/Users/tomkinsm/stomata-air-mattress/cgal/Installation/AUTHORS"
    "/Users/tomkinsm/stomata-air-mattress/cgal/Installation/CHANGES.md"
    "/Users/tomkinsm/stomata-air-mattress/cgal/Installation/LICENSE"
    "/Users/tomkinsm/stomata-air-mattress/cgal/Installation/LICENSE.GPL"
    "/Users/tomkinsm/stomata-air-mattress/cgal/Installation/LICENSE.LGPL"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/AABB_tree/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Advancing_front_surface_reconstruction/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Algebraic_foundations/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Algebraic_kernel_d/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Algebraic_kernel_for_circles/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Algebraic_kernel_for_spheres/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Alpha_shapes_2/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Alpha_shapes_3/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Alpha_wrap_3/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Apollonius_graph_2/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Arithmetic_kernel/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Arrangement_on_surface_2/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/BGL/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Barycentric_coordinates_2/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Basic_viewer/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Boolean_set_operations_2/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Bounding_volumes/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Box_intersection_d/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/CGAL_Core/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/CGAL_ImageIO/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/CGAL_ipelets/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Cartesian_kernel/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Circular_kernel_2/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Circular_kernel_3/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Circulator/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Classification/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Combinatorial_map/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Cone_spanners_2/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Convex_decomposition_3/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Convex_hull_2/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Convex_hull_3/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Convex_hull_d/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Distance_2/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Distance_3/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Envelope_2/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Envelope_3/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Filtered_kernel/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Generalized_map/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Generator/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/GraphicsView/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/HalfedgeDS/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Hash_map/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Heat_method_3/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Homogeneous_kernel/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Hyperbolic_triangulation_2/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Inscribed_areas/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Installation/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Interpolation/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Intersections_2/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Intersections_3/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Interval_skip_list/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Interval_support/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Jet_fitting_3/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Kernel_23/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Kernel_d/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Kinetic_space_partition/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Kinetic_surface_reconstruction/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/LEDA/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Linear_cell_complex/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Matrix_search/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Mesh_2/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Mesh_3/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Mesher_level/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Minkowski_sum_2/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Minkowski_sum_3/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Modifier/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Modular_arithmetic/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Nef_2/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Nef_3/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Nef_S2/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/NewKernel_d/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Number_types/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Optimal_bounding_box/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Optimal_transportation_reconstruction_2/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Optimisation_basic/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Orthtree/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Partition_2/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Periodic_2_triangulation_2/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Periodic_3_mesh_3/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Periodic_3_triangulation_3/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Periodic_4_hyperbolic_triangulation_2/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Point_set_2/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Point_set_3/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Point_set_processing_3/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Poisson_surface_reconstruction_3/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Polygon/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Polygon_mesh_processing/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Polygon_repair/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Polygonal_surface_reconstruction/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Polyhedron/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Polyline_simplification_2/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Polynomial/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Polytope_distance_d/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Principal_component_analysis/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Principal_component_analysis_LGPL/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Profiling_tools/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Property_map/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/QP_solver/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Random_numbers/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Ridges_3/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/SMDS_3/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/STL_Extension/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Scale_space_reconstruction_3/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/SearchStructures/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Segment_Delaunay_graph_2/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Segment_Delaunay_graph_Linf_2/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Set_movable_separability_2/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Shape_detection/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Shape_regularization/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Skin_surface_3/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Snap_rounding_2/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Solver_interface/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Spatial_searching/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Spatial_sorting/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Straight_skeleton_2/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Straight_skeleton_extrusion_2/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Stream_lines_2/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Stream_support/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Subdivision_method_3/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Surface_mesh/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Surface_mesh_approximation/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Surface_mesh_deformation/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Surface_mesh_parameterization/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Surface_mesh_segmentation/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Surface_mesh_shortest_path/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Surface_mesh_simplification/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Surface_mesh_skeletonization/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Surface_mesh_topology/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Surface_mesher/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Surface_sweep_2/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/TDS_2/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/TDS_3/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Testsuite/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Tetrahedral_remeshing/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Three/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Triangulation/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Triangulation_2/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Triangulation_3/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Triangulation_on_sphere_2/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Union_find/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Visibility_2/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Voronoi_diagram_2/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Weights/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/build/include/CGAL" REGEX "/\\.svn$" EXCLUDE REGEX "/qt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/CGAL" TYPE DIRECTORY FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Installation/cmake/modules/")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/CGAL" TYPE FILE FILES
    "/Users/tomkinsm/stomata-air-mattress/cgal/Installation/lib/cmake/CGAL/CGALConfig.cmake"
    "/Users/tomkinsm/stomata-air-mattress/cgal/Installation/lib/cmake/CGAL/CGALConfigBuildVersion.cmake"
    "/Users/tomkinsm/stomata-air-mattress/cgal/Installation/lib/cmake/CGAL/CGALConfigVersion.cmake"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/man/man1" TYPE FILE FILES "/Users/tomkinsm/stomata-air-mattress/cgal/Installation/auxiliary/cgal_create_cmake_script.1")
endif()

