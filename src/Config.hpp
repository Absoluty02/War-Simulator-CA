#pragma once

#include <allegro5/allegro5.h>
#include <fstream> //ifstream
#include <iostream> //cout
#include <assert.h>
#include "mpi.h"

#include "json.hpp"
using json = nlohmann::json;


struct rgb { int r; int g; int b; };

struct Config
{
	bool show_graphics = true;

	bool is_parallel = true;

	// seed used for random number generation
	// 0 means random seed
	int rand_seed = 0;

	// program stops after this many generations
	// 0 means no limit
	int last_generation = 100;

	// % of cells to be walls (0-100) (default: 51)
	int initial_fill_perc = 51;

	uint8_t neighbour_radius = 1;
	int roughness = 1;

	int cols = 480;
	int rows = 360;

	// PARALLEL ONLY
	// number of threads per column
	int x_threads = 1;

	// PARALLEL ONLY
	// number of threads per row
	int y_threads = 1;

	// GRAPHIC ONLY
	int cell_width = 3;

	// GRAPHIC ONLY
	int cell_height = 3;

	// GRAPHIC ONLY
	bool draw_edges = true;

	//GRAPHIC AND PARALLEL ONLY
	bool draw_threads_grid = true;

	// GRAPHIC ONLY
	int max_frame_rate = 60;

	// GRAPHIC ONLY
	rgb wall_color{ 0, 0, 0 };

	// GRAPHIC ONLY
	rgb floor_color{ 255, 255, 255 };

	// GRAPHIC ONLY
	rgb threads_grid_color{ 0, 255, 0 };

	std::string results_file_path = "";


	// Config() : Config("./config/default.cfg") {}

	Config(std::string configFile) {

		std::ifstream config(configFile);
		if(!config.is_open()) {
			std::cout << "Failed to open config file: " << configFile << std::endl;
			std::cout << "Continue with default configuration? (y/n)" << std::endl;
			char c;
			std::cin >> c;
			if(c == 'y') {
				return;
			}
			else {
				std::cout << "Exiting..." << std::endl;
				MPI_Abort(MPI_COMM_WORLD, 1);
				exit(1);
			}
		}

		json jsonConfig;
		config >> jsonConfig;

		if(jsonConfig.contains("show_graphics")) show_graphics = jsonConfig["show_graphics"];

		if(jsonConfig.contains("is_parallel")) is_parallel = jsonConfig["is_parallel"];


		if(jsonConfig.contains("rand_seed")) rand_seed = jsonConfig["rand_seed"];
		if(jsonConfig.contains("last_generation")) last_generation = jsonConfig["last_generation"];

		if(jsonConfig.contains("cols")) cols = jsonConfig["cols"];
		if(jsonConfig.contains("rows")) rows = jsonConfig["rows"];

		if(jsonConfig.contains("x_threads")) x_threads = jsonConfig["x_threads"];
		if(jsonConfig.contains("y_threads")) y_threads = jsonConfig["y_threads"];

		if(jsonConfig.contains("cell_size")) cell_width = cell_height = jsonConfig["cell_size"];
		if(jsonConfig.contains("cell_width")) cell_width = jsonConfig["cell_width"];
		if(jsonConfig.contains("cell_height")) cell_height = jsonConfig["cell_height"];

		if(jsonConfig.contains("draw_edges")) draw_edges = jsonConfig["draw_edges"];
		if(jsonConfig.contains("max_frame_rate")) max_frame_rate = jsonConfig["max_frame_rate"];

		if(jsonConfig.contains("initial_fill_perc")) initial_fill_perc = jsonConfig["initial_fill_perc"];

		if(jsonConfig.contains("neighbour_radius")) neighbour_radius = jsonConfig["neighbour_radius"];
		if(jsonConfig.contains("roughness")) roughness = jsonConfig["roughness"];

		if(jsonConfig.contains("draw_threads_grid")) draw_threads_grid = jsonConfig["draw_threads_grid"];

		if(jsonConfig.contains("wall_color")) {
			wall_color.r = jsonConfig["wall_color"][0];
			wall_color.g = jsonConfig["wall_color"][1];
			wall_color.b = jsonConfig["wall_color"][2];
		}

		if(jsonConfig.contains("floor_color")) {
			floor_color.r = jsonConfig["floor_color"][0];
			floor_color.g = jsonConfig["floor_color"][1];
			floor_color.b = jsonConfig["floor_color"][2];
		}

		if(jsonConfig.contains("threads_grid_color")) {
			threads_grid_color.r = jsonConfig["threads_grid_color"][0];
			threads_grid_color.g = jsonConfig["threads_grid_color"][1];
			threads_grid_color.b = jsonConfig["threads_grid_color"][2];
		}

		if(jsonConfig.contains("results_file_path")) results_file_path = jsonConfig["results_file_path"];

	}
};
