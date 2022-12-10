#include <iostream>
#include <cstdint>
#include <cstdlib>
#include <chrono>
#include <random>
#include <Omega_h_file.hpp>
#include <Omega_h_bbox.hpp>
#include <Omega_h_library.hpp>
#include <Kokkos_Core.hpp>
#include <wdmcpl/point_search.h>

/*
argv[1]: mesh file name
argv[2]: grid dim
argv[3]: n_points = 256
argv[4]: random_seed = 42
*/
int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  {
    static constexpr std::size_t default_n_points = 256;
    static constexpr std::uint_fast32_t default_rand_seed = 42;

    if (argc < 3)
      std::exit(1);
    Omega_h::Library lib{};
    int grid_dim{std::atoi(argv[2])};
    Omega_h::Mesh mesh{&lib};
    Omega_h::binary::read(argv[1], lib.world(), &mesh);
    auto n_points = argc < 4 ? default_n_points : std::atoi(argv[3]);
    auto random_seed = argc < 5 ? default_rand_seed : std::atoi(argv[4]);

    auto bbox = Omega_h::get_bounding_box<2>(&mesh);
    std::mt19937 gen{random_seed};
    std::uniform_real_distribution<> random_x{bbox.min[0], bbox.max[0]},
                                    random_y{bbox.min[1], bbox.max[1]};
    Kokkos::View<wdmcpl::Real *[2]> points("test_points", n_points);
    auto points_h = Kokkos::create_mirror_view(points);
    for (std::size_t i = 0; i < n_points; ++i) {
      points_h(i, 0) = random_x(gen);
      points_h(i, 1) = random_y(gen);
    }

    using std::chrono::steady_clock;
    steady_clock::time_point t[5];
    // start
    t[0] = steady_clock::now();
      wdmcpl::GridPointSearch search{mesh, grid_dim, grid_dim};
    // search structure construction end
    t[1] = steady_clock::now();
      Kokkos::deep_copy(points, points_h);
    // points copy to device end
    t[2] = steady_clock::now();
      auto results = search(points);
    // search end
    t[3] = steady_clock::now();
      auto results_h = Kokkos::create_mirror_view(results);
      Kokkos::deep_copy(results_h, results);
    // points copy to host end
    t[4] = steady_clock::now();

    std::cout << "Timing done.\n";
    auto search_structure_construction_time = t[1] - t[0];
    auto points_copy_time = (t[2] - t[1]) + (t[4] - t[3]);
    auto search_time = t[3] - t[2];

    std::cout << search_structure_construction_time.count() << '\n'
      << points_copy_time.count() << '\n'
      << search_time.count() << std::endl;
  }
  Kokkos::finalize();
}
