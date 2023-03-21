#include <chrono>
#include <iostream>
#include <time.h>
#include <string>

#include <allegro5/allegro.h>
#include <allegro5/allegro_font.h>
#include <allegro5/allegro_ttf.h>
#include <allegro5/allegro_primitives.h>
#include <algorithm>
#include <mpi.h>
#include <random>
#include "Config.hpp"

#define ROOT_RANK 0
#define BEINGBEATENCHANCE 10

Config *cfg;
std::string config_file_path = "./config/default.cfg";

union Person
{
	struct
	{
		uint16_t defender : 1;
		uint16_t attacker : 1;
		uint16_t originalPrey : 1;
		uint16_t originalPredator : 1;
		uint16_t isChild : 1;
		uint16_t isDead : 1;
		uint16_t time : 3;
	} inside;

	uint16_t all;
};

Person *writeMatrix;
Person *readMatrix;
Person *root_grid; // full grid used for drawing

int my_rank = 0; // MPI rank
int n_procs = 1; // MPI size

int my_rows, my_inner_rows, tot_inner_rows;
int my_cols, my_inner_cols, tot_inner_cols;

int inner_grid_size;
int outer_grid_size;

int generation = 0;
bool is_running = true;

double total_time = 0;
double communication_time = 0;
double generation_time = 0;
double draw_time = 0;
double start_time, end_time;
// double* frame_times;

int neighbours_ranks[3][3];

MPI_Datatype innerGridType;
MPI_Datatype contiguousGridType;
MPI_Datatype columnType; // for sending/receiving left and right columns
MPI_Datatype rowType;	   // for sending/receiving top and bottom rows
MPI_Datatype cornerType; // for sending/receiving corners
MPI_Comm myComm;

ALLEGRO_FONT *font;
ALLEGRO_DISPLAY *display;
ALLEGRO_EVENT_QUEUE *queue;
ALLEGRO_TIMER *timer;

ALLEGRO_COLOR wall_color, floor_color;
int DISPLAY_WIDTH, DISPLAY_HEIGHT;

ALLEGRO_COLOR threads_grid_color;

inline int at(int y, int x)
{
	return y * my_cols + x;
}

inline bool isValidCoord(int y, int x)
{
	return y >= 0 && y < cfg->y_threads && x >= 0 && x < cfg->x_threads;
}
inline bool isValidCoord(const int coords[])
{
	return isValidCoord(coords[0], coords[1]);
}
inline bool isValidCoord(int coords[])
{
	return isValidCoord(coords[0], coords[1]);
}

inline bool isCoordsEqual(const int coords1[], const int coords2[])
{
	return coords1[0] == coords2[0] && coords1[1] == coords2[1];
}
inline bool isCoordsEqual(int coords1[], int coords2[])
{
	return coords1[0] == coords2[0] && coords1[1] == coords2[1];
}

void initialize(int argc, char const *argv[]);
void serial_initialize_random_grid();
void terminate();
void frame_update();
void update_grid();
void no_graphic_loop();

void write_header(std::ofstream &file);
void write_result(std::ofstream &file);
void end_recap();

void get_config_file_path(int argc, char const *argv[]);
void get_arg_configs(int argc, char const *argv[]);

void print_help();
void print_config_help();

// graphic only
void graphic_initialize();
void serial_draw_grid();
void check_graphic_settings();
void graphic_serial_loop();
void graphic_parallel_loop();

// parallel only
void parallel_draw_grid();
void parallel_initialize_random_grid();
void parallel_initialize();
void check_parallel_settings();

void scatter_initial_grid();
void gather_grid();

void send_columns();
void send_rows();
void send_corners();

void receive_columns();
void receive_rows();
void receive_corners();

inline void exit();

/*
 * ==================================================================================
 *  --------------------------------------------------------------------------------
 *  									MAIN
 *  --------------------------------------------------------------------------------
 * ==================================================================================
 */

int main(int argc, char const *argv[])
{
	initialize(argc, argv);

	start_time = MPI_Wtime();

	if (cfg->show_graphics)
	{
		if (cfg->is_parallel)
			graphic_parallel_loop();
		else
			graphic_serial_loop();
	}
	else
		no_graphic_loop();

	end_time = MPI_Wtime();
	total_time = end_time - start_time;
	end_recap();

	terminate();
	return 0;
}

/*
 * ==================================================================================
 *  --------------------------------------------------------------------------------
 *  								INITIALIZE
 *  --------------------------------------------------------------------------------
 * ==================================================================================
 */

void initialize(int argc, char const *argv[])
{
	get_config_file_path(argc, argv);

	cfg = new Config(config_file_path);
	get_arg_configs(argc, argv);

	tot_inner_rows = cfg->rows;
	tot_inner_cols = cfg->cols;

	// frame_times = new double[cfg->last_generation];

	// checkGeneralSettings();

	if (cfg->is_parallel)
	{
		my_inner_rows = cfg->rows / cfg->y_threads;
		my_inner_cols = cfg->cols / cfg->x_threads;
	}
	else
	{
		my_inner_rows = cfg->rows;
		my_inner_cols = cfg->cols;
	}

	my_rows = my_inner_rows + 2;
	my_cols = my_inner_cols + 2;
	inner_grid_size = my_inner_rows * my_inner_cols;
	outer_grid_size = my_rows * my_cols;

	MPI_Init(NULL, NULL);
	if (cfg->is_parallel)
		parallel_initialize();

	if (cfg->show_graphics)
		graphic_initialize();

	writeMatrix = new Person[outer_grid_size];
	readMatrix = new Person[outer_grid_size];
	// std::fill_n(readMatrix, outer_grid_size, 1(Person));
	for (int i = 0; i < outer_grid_size; ++i)
	{
		readMatrix[i].all = 0;
	}

	if (cfg->is_parallel)
	{
		if (my_rank == ROOT_RANK)
		{
			parallel_initialize_random_grid();
		}
		scatter_initial_grid();
	}
	else
		serial_initialize_random_grid();

	// std::copy_n(readMatrix, outer_grid_size, writeMatrix);
	for (int i = 0; i < outer_grid_size; ++i)
	{
		writeMatrix[i] = readMatrix[i];
	}
}

void graphic_initialize()
{
	if (my_rank == ROOT_RANK)
	{
		check_graphic_settings();
		DISPLAY_WIDTH = cfg->cols * cfg->cell_width;
		DISPLAY_HEIGHT = cfg->rows * cfg->cell_height;
		if (cfg->draw_edges)
		{
			DISPLAY_WIDTH += 2 * cfg->cell_width;
			DISPLAY_HEIGHT += 2 * cfg->cell_height;
		}

		wall_color = al_map_rgb(cfg->wall_color.r, cfg->wall_color.g, cfg->wall_color.b);
		floor_color = al_map_rgb(cfg->floor_color.r, cfg->floor_color.g, cfg->floor_color.b);
		threads_grid_color = al_map_rgb(cfg->threads_grid_color.r, cfg->threads_grid_color.g, cfg->threads_grid_color.b);
		if (!al_init())
			fprintf(stderr, "Failed to initialize allegro.\n");
		if (!al_install_keyboard())
			fprintf(stderr, "Failed to install keyboard.\n");
		if (!al_init_font_addon())
			fprintf(stderr, "Failed to initialize font addon.\n");
		if (!al_init_ttf_addon())
			fprintf(stderr, "Failed to initialize ttf addon.\n");
		if (!al_init_primitives_addon())
			fprintf(stderr, "Failed to initialize primitives addon.\n");

		font = al_load_ttf_font("fonts/Roboto/Roboto-Medium.ttf", 20, 0);
		display = al_create_display(DISPLAY_WIDTH, DISPLAY_HEIGHT);
		queue = al_create_event_queue();
		timer = al_create_timer(1.0 / cfg->max_frame_rate);

		if (!font)
			fprintf(stderr, "Failed load font.\n");
		if (!display)
			fprintf(stderr, "Failed initialize display.\n");
		if (!queue)
			fprintf(stderr, "Failed create queue.\n");
		if (!timer)
			fprintf(stderr, "Failed to create timer.\n");

		al_register_event_source(queue, al_get_display_event_source(display));
		al_register_event_source(queue, al_get_keyboard_event_source());
		al_register_event_source(queue, al_get_timer_event_source(timer));
		al_start_timer(timer);
		al_start_timer(timer);

		al_set_window_title(display, "War Simulator - Jacopo Garofalo 220557");
	}
}

/*
 * ==================================================================================
 *  --------------------------------------------------------------------------------
 *  							  PARALLEL INITIALIZE
 *  --------------------------------------------------------------------------------
 * ==================================================================================
 */
void parallel_initialize()
{

	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &n_procs);

	check_parallel_settings();

	int dims[2] = {cfg->y_threads, cfg->x_threads};
	int periods[2] = {0, 0};

	MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &myComm);

	int my_coords[2];
	MPI_Cart_coords(myComm, my_rank, 2, my_coords);

	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			const int coords[] = {my_coords[0] + i - 1, my_coords[1] + j - 1};
			if (isValidCoord(coords))
			{
				MPI_Cart_rank(myComm, coords, &neighbours_ranks[i][j]);
			}
			else
			{
				neighbours_ranks[i][j] = MPI_PROC_NULL; // -1
			}
		}
	}

	const int outer_sizes[] = {my_rows, my_cols};
	const int inner_sizes[] = {my_inner_rows, my_inner_cols};
	const int starts[] = {0, 0};
	MPI_Type_create_subarray(2, outer_sizes, inner_sizes, starts, MPI_ORDER_C, MPI_UINT16_T, &innerGridType);
	MPI_Type_contiguous(inner_grid_size, MPI_UINT16_T, &contiguousGridType);

	MPI_Type_vector(my_inner_rows, 1, my_cols, MPI_UINT16_T, &columnType);
	MPI_Type_vector(1, my_inner_cols, my_cols, MPI_UINT16_T, &rowType);
	MPI_Type_vector(1, 1, my_cols, MPI_UINT16_T, &cornerType);

	MPI_Type_commit(&innerGridType);
	MPI_Type_commit(&contiguousGridType);
	MPI_Type_commit(&columnType);
	MPI_Type_commit(&rowType);
	MPI_Type_commit(&cornerType);
}

void serial_initialize_random_grid()
{
	if (cfg->rand_seed)
		srand(cfg->rand_seed);
	else
	{
		int seed = time(NULL);
		srand(seed);
		std::cout << "Random seed: " << seed << std::endl;
	}

	int fill_perc = cfg->initial_fill_perc;
	for (int i = 0; i < my_rows; i++)
	{
		for (int j = 0; j < my_cols; j++)
		{
			int chance = rand() % 100 + 1;
			if (chance <= 70)
			{
				readMatrix[at(i, j)].inside.originalPredator = 1;
				readMatrix[at(i, j)].inside.attacker = 1;
			}
			else
			{
				readMatrix[at(i, j)].inside.originalPrey = 1;
				readMatrix[at(i, j)].inside.defender = 1;
			}

			int grownChance = rand() % 2;
			if (grownChance == 0)
			{
				readMatrix[at(i, j)].inside.isChild = 1;
				readMatrix[at(i, j)].inside.time = rand() % 7;
			}
			else
			{
				readMatrix[at(i, j)].inside.isChild = 0;
			}

			writeMatrix[at(i, j)].all = readMatrix[at(i, j)].all;
		}
	}
}

void terminate()
{
	if (my_rank == ROOT_RANK && cfg->show_graphics)
	{
		al_uninstall_keyboard();
		al_destroy_event_queue(queue);
		al_destroy_display(display);
		al_destroy_font(font);
	}

	// delete[] frame_times;

	if (cfg->is_parallel)
	{
		if (my_rank == ROOT_RANK)
		{
			delete[] root_grid;
		}

		MPI_Type_free(&innerGridType);
		MPI_Type_free(&contiguousGridType);
		MPI_Type_free(&columnType);
		MPI_Type_free(&rowType);
		MPI_Type_free(&cornerType);

		MPI_Comm_free(&myComm);
	}
	MPI_Finalize();

	delete cfg;
	delete[] readMatrix;
	delete[] writeMatrix;
}

void check_general_settings()
{
}

void check_parallel_settings()
{
	if (cfg->x_threads < 1 || cfg->y_threads < 1)
	{
		std::cout << "x_threads and y_threads must be at least 1" << std::endl;
		exit();
	}
	if (cfg->rows % cfg->y_threads != 0)
	{
		std::cout << "Rows have to be divisible by the y_threads" << std::endl;
		std::cout << "rows: " << cfg->rows << " y_threads: " << cfg->y_threads << std::endl;
		exit();
	}
	if (cfg->cols % cfg->x_threads != 0)
	{
		std::cout << "Cols have to be divisible by the x_threads" << std::endl;
		std::cout << "cols: " << cfg->cols << " x_threads: " << cfg->x_threads << std::endl;
		exit();
	}
	if (cfg->y_threads * cfg->x_threads != n_procs)
	{
		// make sure the product of dims is equal to the number of processes
		std::cout << "Error: number of processes does not match the number of threads" << std::endl;
		std::cout << "number of processes: " << n_procs << std::endl;
		std::cout << "number of threads per row: " << cfg->y_threads << std::endl;
		std::cout << "number of threads per column: " << cfg->x_threads << std::endl;
		exit();
	}
}

void check_graphic_settings()
{
	if (cfg->cols * cfg->rows > 1382400)
	{
		std::cout << "Grid is too large for graphic mode" << std::endl;
		exit();
	}
}

/*
 * ==================================================================================
 *  --------------------------------------------------------------------------------
 *  									MAIN LOOP
 *  --------------------------------------------------------------------------------
 * ==================================================================================
 */

void graphic_parallel_loop()
{
	while (is_running)
	{
		if (my_rank == ROOT_RANK)
		{
			ALLEGRO_EVENT event;
			al_wait_for_event(queue, &event);
			if (event.type == ALLEGRO_EVENT_KEY_UP || event.type == ALLEGRO_EVENT_DISPLAY_CLOSE)
			{
				std::cout << "Abortings..." << std::endl;
				exit();
			}
			else if (event.type == ALLEGRO_EVENT_TIMER)
			{
				frame_update();
				if (++generation == cfg->last_generation)
				{
					is_running = false;
				}
			}
		}
		else
		{
			frame_update();
			if (++generation == cfg->last_generation)
			{
				is_running = false;
			}
		}
	}
}

void graphic_serial_loop()
{
	while (is_running)
	{
		ALLEGRO_EVENT event;
		al_wait_for_event(queue, &event);
		if (event.type == ALLEGRO_EVENT_KEY_UP || event.type == ALLEGRO_EVENT_DISPLAY_CLOSE)
		{
			std::cout << "Abortings..." << std::endl;
			exit();
		}
		else if (event.type == ALLEGRO_EVENT_TIMER)
		{
			frame_update();
			if (++generation == cfg->last_generation)
			{
				is_running = false;
			}
		}
	}
}

void no_graphic_loop()
{
	while (is_running)
	{
		frame_update();
		//printf("%d\n", generation);

		if (++generation == cfg->last_generation)
		{
			is_running = false;
		}
	}
}

void frame_update()
{
	// double frame_start_time = MPI_Wtime();
	if (cfg->show_graphics)
	{
		if (my_rank == ROOT_RANK)
		{
			double start_draw_time = MPI_Wtime();
			al_flip_display();
			al_clear_to_color(wall_color);

			if (cfg->is_parallel)
			{
				// receive the grid from the other processes
				double receive_start_time = MPI_Wtime();
				gather_grid();
				communication_time += MPI_Wtime() - receive_start_time;

				parallel_draw_grid();
			}
			else
			{
				serial_draw_grid();
			}

			double end_draw_time = MPI_Wtime();
			draw_time += end_draw_time - start_draw_time;
		}
		else
		{ // my_rank != ROOT_RANK

			// send read_grid to root
			double receive_start_time = MPI_Wtime();
			gather_grid();
			communication_time += MPI_Wtime() - receive_start_time;
		}
	}

	if (cfg->is_parallel)
	{
		// send columns to other processes
		double comms_start_time = MPI_Wtime();

		send_columns();

		// send rows to other processes
		send_rows();

		// send corners to other processes
		send_corners();

		// receive columns from other processes
		receive_columns();

		// receive rows from other processes
		receive_rows();

		// receive corners from other processes
		receive_corners();

		communication_time += MPI_Wtime() - comms_start_time;
	}

	double generation_start_time = MPI_Wtime();
	update_grid();
	generation_time += MPI_Wtime() - generation_start_time;
	std::swap(readMatrix, writeMatrix);

	// double frame_end_time = MPI_Wtime();
	// frame_times[generation] = frame_end_time - frame_start_time;
}


void update_grid()
{
	for (int i = 1; i < my_rows - 1; i++)
	{
		for (int j = 1; j < my_cols - 1; j++)
		{
			if (readMatrix[at(i, j)].inside.isDead)
			{

				writeMatrix[at(i, j)] = readMatrix[at(i, j)];
				if (readMatrix[at(i, j)].inside.time == 7)
				{
					writeMatrix[at(i, j)].all = 0;
					
					int race = rand() % 2;
					if (race == 1)
					{
						writeMatrix[at(i, j)].inside.originalPredator = 1;
						writeMatrix[at(i, j)].inside.attacker = 1;
					}
					else
					{
						writeMatrix[at(i, j)].inside.originalPrey = 1;
						writeMatrix[at(i, j)].inside.defender = 1;
					}
					writeMatrix[at(i, j)].inside.isChild = 1;
				}
				else if (readMatrix[at(i, j)].inside.time < 7)
				{
					++writeMatrix[at(i, j)].inside.time;
				}
				continue;
			}
			else if (readMatrix[at(i, j)].inside.defender && !readMatrix[at(i, j)].inside.isDead)
			{

				int probability = 0;

				if (readMatrix[at(i, j)].inside.isChild == 1)
					probability = 2;

				for (int k = -1; k <= 1; ++k)
				{

					for (int l = -1; l <= 1; ++l)
					{

						if (readMatrix[at(i + k, j + l)].inside.isChild && readMatrix[at(i + k, j + l)].inside.attacker)
						{
							++probability;
						}
						else if (!readMatrix[at(i + k, j + l)].inside.isChild && readMatrix[at(i + k, j + l)].inside.attacker)
						{
							probability += 2;
						}
						else if (readMatrix[at(i + k, j + l)].inside.isChild && !readMatrix[at(i + k, j + l)].inside.attacker)
						{
							--probability;
						}
						else if (!readMatrix[at(i + k, j + l)].inside.isChild && !readMatrix[at(i + k, j + l)].inside.attacker)
						{
							probability -= 2;
						}
					}
				}

				writeMatrix[at(i, j)] = readMatrix[at(i, j)];
				if (!readMatrix[at(i, j)].inside.isChild)
				{
					if (readMatrix[at(i, j)].inside.time == 7)
					{
						writeMatrix[at(i, j)].all = 0;
						writeMatrix[at(i, j)].inside.isDead = 1;
						writeMatrix[at(i, j)].inside.time = 0;
					}
					else if (readMatrix[at(i, j)].inside.time < 7)
					{
						++writeMatrix[at(i, j)].inside.time;
					}
				}
				if (readMatrix[at(i, j)].inside.isDead || probability <= 0)
				{

					if (readMatrix[at(i, j)].inside.time == 7 && readMatrix[at(i, j)].inside.isChild)
					{
						writeMatrix[at(i, j)].inside.isChild = 0;
						writeMatrix[at(i, j)].inside.time = 0;
					}
					else if (readMatrix[at(i, j)].inside.time < 7 && readMatrix[at(i, j)].inside.isChild)
					{
						++writeMatrix[at(i, j)].inside.time;
					}

					continue;
				}

				else
				{
					if (16 / probability < BEINGBEATENCHANCE)
					{
						writeMatrix[at(i, j)].all = 0;
						writeMatrix[at(i, j)].inside.isDead = 1;
					}
					else
					{
						if (readMatrix[at(i, j)].inside.time == 7 && readMatrix[at(i, j)].inside.isChild)
						{
							writeMatrix[at(i, j)].inside.isChild = 0;
							writeMatrix[at(i, j)].inside.time = 0;
						}
						else if (readMatrix[at(i, j)].inside.time < 7 && readMatrix[at(i, j)].inside.isChild)
						{
							++writeMatrix[at(i, j)].inside.time;
						}
					}
				}
			}

			else if (readMatrix[at(i, j)].inside.attacker == 1)
			{

				int probability = 0;

				if (readMatrix[at(i, j)].inside.isChild == 1)
					probability = 2;

				for (int k = -1; k <= 1; ++k)
				{

					for (int l = -1; l <= 1; ++l)
					{

						if (readMatrix[at(i + k, j + l)].inside.isChild && readMatrix[at(i + k, j + l)].inside.attacker)
						{
							--probability;
						}
						else if (!readMatrix[at(i + k, j + l)].inside.isChild && readMatrix[at(i + k, j + l)].inside.attacker)
						{
							probability -= 2;
						}
						else if (readMatrix[at(i + k, j + l)].inside.isChild && !readMatrix[at(i + k, j + l)].inside.attacker)
						{
							++probability;
						}
						else if (!readMatrix[at(i + k, j + l)].inside.isChild && !readMatrix[at(i + k, j + l)].inside.attacker)
						{
							probability += 2;
						}
					}
				}
				writeMatrix[at(i, j)] = readMatrix[at(i, j)];
				if (!readMatrix[at(i, j)].inside.isChild)
				{
					if (readMatrix[at(i, j)].inside.time == 7)
					{
						writeMatrix[at(i, j)].all = 0;
						writeMatrix[at(i, j)].inside.isDead = 1;
						writeMatrix[at(i, j)].inside.time = 0;
					}
					else if (readMatrix[at(i, j)].inside.time < 7)
					{
						writeMatrix[at(i, j)].inside.time++;
					}
				}
				if (readMatrix[at(i, j)].inside.isDead || probability <= 0)
				{
					if (readMatrix[at(i, j)].inside.time == 7 && readMatrix[at(i, j)].inside.isChild)
					{
						writeMatrix[at(i, j)].inside.isChild = 0;
						writeMatrix[at(i, j)].inside.time = 0;
					}
					else if (readMatrix[at(i, j)].inside.time < 7 && readMatrix[at(i, j)].inside.isChild)
					{
						writeMatrix[at(i, j)].inside.time++;
					}
				}
				else
				{
					if (16 / probability < BEINGBEATENCHANCE)
					{
						writeMatrix[at(i, j)].all = 0;
						writeMatrix[at(i, j)].inside.isDead = 1;
					}
					else
					{
						if (readMatrix[at(i, j)].inside.time == 7 && readMatrix[at(i, j)].inside.isChild)
						{
							writeMatrix[at(i, j)].inside.isChild = 0;
							writeMatrix[at(i, j)].inside.time = 0;
						}
						else if (readMatrix[at(i, j)].inside.time < 7 && readMatrix[at(i, j)].inside.isChild)
						{
							writeMatrix[at(i, j)].inside.time++;
						}
					}
				}
			}
		}
	}
}

/*
 * ==================================================================================
 *  --------------------------------------------------------------------------------
 *  									DRAWING
 *  --------------------------------------------------------------------------------
 * ==================================================================================
 */

void parallel_draw_grid()
{
	// offset in cells, not pixels
	int edge_offset = edge_offset = cfg->draw_edges ? 1 : 0;

	for (int proc = 0; proc < n_procs; proc++)
	{

		int proc_x = (proc % cfg->x_threads) * my_inner_cols + edge_offset;
		int proc_y = (proc / cfg->x_threads) * my_inner_rows + edge_offset;

		int proc_x2 = ((proc % cfg->x_threads) + 1) * my_inner_cols + edge_offset;
		int proc_y2 = ((proc / cfg->x_threads) + 1) * my_inner_rows + edge_offset;

		for (int i = 0; i < my_inner_rows; i++)
		{

			int y = (i + proc_y) * cfg->cell_height;

			for (int j = 0; j < my_inner_cols; j++)
			{

				int idx = (proc * inner_grid_size) + (i * my_inner_cols) + j;
				int x = (j + proc_x) * cfg->cell_width;

				if (root_grid[idx].inside.isDead)
				{
					al_draw_filled_rectangle(x, y, x + cfg->cell_width, y + cfg->cell_height, al_map_rgb(0, 0, 0));
				}
				else
				{
					if (root_grid[idx].inside.originalPredator)
					{
						if (root_grid[idx].inside.isChild == 1)
							al_draw_filled_rectangle(x, y, x + cfg->cell_width, y + cfg->cell_height, al_map_rgb(255, 117, 20));
						else
							al_draw_filled_rectangle(x, y, x + cfg->cell_width, y + cfg->cell_height, al_map_rgb(217, 89, 0));
					}
					else if (root_grid[idx].inside.originalPrey)
					{
						if (root_grid[idx].inside.isChild == 1)
							al_draw_filled_rectangle(x, y, x + cfg->cell_width, y + cfg->cell_height, al_map_rgb(255, 255, 255));
						else
							al_draw_filled_rectangle(x, y, x + cfg->cell_width, y + cfg->cell_height, al_map_rgb(156, 156, 156));
					}
				}
			}
		}

		if (cfg->draw_threads_grid)
		{
			al_draw_rectangle(proc_x * cfg->cell_width, proc_y * cfg->cell_height, proc_x2 * cfg->cell_width, proc_y2 * cfg->cell_height, threads_grid_color, 1);
		}
	}
}

void serial_draw_grid()
{

	for (int i = 1; i < my_rows - 1; i++)
	{
		for (int j = 1; j < my_cols - 1; j++)
		{

			int y = (i - (!cfg->draw_edges * 1)) * cfg->cell_height;
			int x = (j - (!cfg->draw_edges * 1)) * cfg->cell_width;

			if (readMatrix[at(i, j)].inside.isDead)
			{
				al_draw_filled_rectangle(x, y, x + cfg->cell_width, y + cfg->cell_height, al_map_rgb(0, 0, 0));
			}
			else
			{
				if (readMatrix[at(i, j)].inside.originalPredator)
				{
					if (readMatrix[at(i, j)].inside.isChild == 1)
						al_draw_filled_rectangle(x, y, x + cfg->cell_width, y + cfg->cell_height, al_map_rgb(255, 117, 20));
					else
						al_draw_filled_rectangle(x, y, x + cfg->cell_width, y + cfg->cell_height, al_map_rgb(217, 89, 0));
				}
				else if (readMatrix[at(i, j)].inside.originalPrey)
				{
					if (readMatrix[at(i, j)].inside.isChild == 1)
						al_draw_filled_rectangle(x, y, x + cfg->cell_width, y + cfg->cell_height, al_map_rgb(255, 255, 255));
					else
						al_draw_filled_rectangle(x, y, x + cfg->cell_width, y + cfg->cell_height, al_map_rgb(156, 156, 156));
				}
			}
		}
	}
}

/*
 * ==================================================================================
 *  --------------------------------------------------------------------------------
 *  							PARALLEL COMUNICATION
 *  --------------------------------------------------------------------------------
 * ==================================================================================
 */

void parallel_initialize_random_grid()
{
	if (cfg->rand_seed)
		srand(cfg->rand_seed);
	else
	{
		int seed = time(NULL);
		srand(seed);
		std::cout << "Random seed: " << seed << std::endl;
	}

	root_grid = new Person[inner_grid_size * n_procs];

	for (int proc = 0; proc < n_procs; proc++)
	{
		for (int i = 0; i < my_inner_rows; i++)
		{
			for (int j = 0; j < my_inner_cols; j++)
			{
				int idx = (proc * inner_grid_size) + (i * my_inner_cols) + j;
				int chance = rand() % 100;
				if (chance <= 40)
				{
					root_grid[idx].inside.originalPredator = 1;
					root_grid[idx].inside.attacker = 1;
				}
				else
				{
					root_grid[idx].inside.originalPrey = 1;
					root_grid[idx].inside.defender = 1;
				}

				int grownChance = rand() % 2;
				if (grownChance == 0)
				{
					root_grid[idx].inside.isChild = 1;
					root_grid[idx].inside.time = rand() % 7;
				}
				else
				{
					root_grid[idx].inside.isChild = 0;
				}
			}
		}
	}
}

void scatter_initial_grid()
{
	// root sends initial grid to all other processes
	Person *dest_buff = &readMatrix[my_cols + 1];
	MPI_Scatter(root_grid, 1, contiguousGridType, dest_buff, 1, innerGridType, ROOT_RANK, myComm);
}

void gather_grid()
{
	Person *send_buff = &readMatrix[my_cols + 1];
	MPI_Gather(send_buff, 1, innerGridType, root_grid, 1, contiguousGridType, ROOT_RANK, myComm);
}

#define TOP 0
#define MIDDLE 1
#define BOTTOM 2
#define LEFT 0
#define RIGHT 2

void send_columns()
{
	if (neighbours_ranks[MIDDLE][LEFT] != MPI_PROC_NULL)
	{
		MPI_Request req;
		int start_idx = my_cols + 1;
		MPI_Isend(&readMatrix[start_idx], 1, columnType, neighbours_ranks[MIDDLE][LEFT], 1001, myComm, &req);
		MPI_Request_free(&req);
	}

	if (neighbours_ranks[MIDDLE][RIGHT] != MPI_PROC_NULL)
	{
		MPI_Request req;
		int start_idx = my_cols + my_inner_cols;
		MPI_Isend(&readMatrix[start_idx], 1, columnType, neighbours_ranks[MIDDLE][RIGHT], 1002, myComm, &req);
		MPI_Request_free(&req);
	}
	// #ifdef DEBUG_MODE
	// std::cout << "[" << my_rank << "]: columns sent" << std::endl;
	// #endif // DEBUG_MODE
}

void send_rows()
{
	if (neighbours_ranks[TOP][MIDDLE] != MPI_PROC_NULL)
	{
		MPI_Request req;
		int start_idx = my_cols + 1;
		MPI_Isend(&readMatrix[start_idx], 1, rowType, neighbours_ranks[TOP][MIDDLE], 1003, myComm, &req);
		MPI_Request_free(&req);
	}

	if (neighbours_ranks[BOTTOM][MIDDLE] != MPI_PROC_NULL)
	{
		MPI_Request req;
		int start_idx = my_cols * my_inner_rows + 1;
		MPI_Isend(&readMatrix[start_idx], 1, rowType, neighbours_ranks[BOTTOM][MIDDLE], 1004, myComm, &req);
		MPI_Request_free(&req);
	}
}

void send_corners()
{
	if (neighbours_ranks[TOP][LEFT] != MPI_PROC_NULL)
	{
		MPI_Request req;
		int start_idx = my_cols + 1;
		MPI_Isend(&readMatrix[start_idx], 1, cornerType, neighbours_ranks[TOP][LEFT], 1005, myComm, &req);
		MPI_Request_free(&req);
	}

	if (neighbours_ranks[TOP][RIGHT] != MPI_PROC_NULL)
	{
		MPI_Request req;
		int start_idx = my_cols + my_inner_cols;
		MPI_Isend(&readMatrix[start_idx], 1, cornerType, neighbours_ranks[TOP][RIGHT], 1006, myComm, &req);
		MPI_Request_free(&req);
	}

	if (neighbours_ranks[BOTTOM][LEFT] != MPI_PROC_NULL)
	{
		MPI_Request req;
		int start_idx = my_cols * my_inner_rows + 1;
		MPI_Isend(&readMatrix[start_idx], 1, cornerType, neighbours_ranks[BOTTOM][LEFT], 1007, myComm, &req);
		MPI_Request_free(&req);
	}

	if (neighbours_ranks[BOTTOM][RIGHT] != MPI_PROC_NULL)
	{
		MPI_Request req;
		int start_idx = my_cols * my_inner_rows + my_inner_cols;
		MPI_Isend(&readMatrix[start_idx], 1, cornerType, neighbours_ranks[BOTTOM][RIGHT], 1008, myComm, &req);
		MPI_Request_free(&req);
	}
}

void receive_columns()
{

	// std::cout << "[" << my_rank << "]: receiving columns" << std::endl;
	if (neighbours_ranks[MIDDLE][RIGHT] != MPI_PROC_NULL)
	{
		int start_idx = my_cols + my_inner_cols + 1;
		MPI_Recv(&readMatrix[start_idx], 1, columnType, neighbours_ranks[MIDDLE][RIGHT], 1001, myComm, MPI_STATUS_IGNORE);
	}
	if (neighbours_ranks[MIDDLE][LEFT] != MPI_PROC_NULL)
	{
		int start_idx = my_cols;
		MPI_Recv(&readMatrix[start_idx], 1, columnType, neighbours_ranks[MIDDLE][LEFT], 1002, myComm, MPI_STATUS_IGNORE);
	}
	// std::cout << "[" << my_rank << "]: columns received" << std::endl;
}

void receive_rows()
{

	if (neighbours_ranks[BOTTOM][MIDDLE] != MPI_PROC_NULL)
	{
		int start_idx = my_cols * (my_rows - 1) + 1;
		MPI_Recv(&readMatrix[start_idx], 1, rowType, neighbours_ranks[BOTTOM][MIDDLE], 1003, myComm, MPI_STATUS_IGNORE);
	}

	if (neighbours_ranks[TOP][MIDDLE] != MPI_PROC_NULL)
	{
		int start_idx = 1;
		MPI_Recv(&readMatrix[start_idx], 1, rowType, neighbours_ranks[TOP][MIDDLE], 1004, myComm, MPI_STATUS_IGNORE);
	}
}

void receive_corners()
{
	if (neighbours_ranks[BOTTOM][RIGHT] != MPI_PROC_NULL)
	{
		int start_idx = my_cols * (my_inner_rows + 1) + my_inner_cols + 1;
		MPI_Recv(&readMatrix[start_idx], 1, cornerType, neighbours_ranks[BOTTOM][RIGHT], 1005, myComm, MPI_STATUS_IGNORE);
	}
	if (neighbours_ranks[BOTTOM][LEFT] != MPI_PROC_NULL)
	{
		int start_idx = my_cols * (my_inner_rows + 1);
		MPI_Recv(&readMatrix[start_idx], 1, cornerType, neighbours_ranks[BOTTOM][LEFT], 1006, myComm, MPI_STATUS_IGNORE);
	}
	if (neighbours_ranks[TOP][RIGHT] != MPI_PROC_NULL)
	{
		int start_idx = my_inner_cols + 1;
		MPI_Recv(&readMatrix[start_idx], 1, cornerType, neighbours_ranks[TOP][RIGHT], 1007, myComm, MPI_STATUS_IGNORE);
	}
	if (neighbours_ranks[TOP][LEFT] != MPI_PROC_NULL)
	{
		int start_idx = 0;
		MPI_Recv(&readMatrix[start_idx], 1, cornerType, neighbours_ranks[TOP][LEFT], 1008, myComm, MPI_STATUS_IGNORE);
	}
}

void write_header(std::ofstream &file)
{
	std::string separator = ",";
	file << "total_time" << separator
		 << "communication_time" << separator
		 << "generation_time" << separator
		 << "draw_time" << separator
		 << "start_time" << separator
		 << "end_time" << separator
		 << "show_graphics" << separator
		 << "is_parallel" << separator
		 << "n_procs" << separator
		 << "x_threads" << separator
		 << "y_threads" << separator
		 << "cols" << separator
		 << "rows" << separator
		 << "1" << separator
		 << "config_file_path" << std::endl;
}


void write_result(std::ofstream &file)
{
	std::string separator = ",";
	file << total_time << separator
		 << communication_time << separator
		 << generation_time << separator
		 << draw_time << separator
		 << start_time << separator
		 << end_time << separator
		 << cfg->show_graphics << separator
		 << cfg->is_parallel << separator
		 << n_procs << separator
		 << cfg->x_threads << separator
		 << cfg->y_threads << separator
		 << cfg->cols << separator
		 << cfg->rows << separator
		 << config_file_path << std::endl;
	// print_frame_times(file);

	// file << std::endl;
}

void get_config_file_path(int argc, char const *argv[])
{
	for (int i = 0; i < argc; i++)
	{
		if ((argv[i] == std::string("-c") || argv[i] == std::string("--config")) && i + 1 < argc)
		{
			config_file_path = argv[++i];
			return;
		}
	}
}

void end_recap()
{
	if (my_rank != ROOT_RANK)
		return;

	std::cout << std::endl;
	std::cout << "Communication time: " << communication_time << " s" << std::endl;
	std::cout << "Generation time:    " << generation_time << " s" << std::endl;
	std::cout << "Draw time:          " << draw_time << " s" << std::endl;
	std::cout << "Total time:         " << total_time << " s" << std::endl;

	if (cfg->results_file_path.empty())
	{
		std::cout << "No results file path specified" << std::endl;
	}
	else
	{
		std::ofstream file;
		if (std::filesystem::exists(cfg->results_file_path))
		{
			file.open(cfg->results_file_path, std::ios::app);
		}
		else
		{
			std::cout << "Creating results file" << std::endl;
			file.open(cfg->results_file_path);
			write_header(file);
		}
		write_result(file);
		file.close();

		std::cout << "Results written to " << cfg->results_file_path << std::endl;
	}
}

void get_arg_configs(int argc, char const *argv[])
{
	for (int i = 0; i < argc; i++)
	{
		if (argv[i] == std::string("-h") || argv[i] == std::string("--help"))
		{
			print_help();
			exit();
		}
		if (argv[i] == std::string("-hc") || argv[i] == std::string("--help-config"))
		{
			print_config_help();
			exit();
		}
		else if (argv[i] == std::string("-g") || argv[i] == std::string("--graphic"))
		{
			cfg->show_graphics = true;
		}
		else if (argv[i] == std::string("-G") || argv[i] == std::string("--no-graphic"))
		{
			cfg->show_graphics = false;
		}
		else if (argv[i] == std::string("-p") || argv[i] == std::string("--parallel"))
		{
			cfg->is_parallel = true;
		}
		else if (argv[i] == std::string("-P") || argv[i] == std::string("-s") || argv[i] == std::string("--serial"))
		{
			cfg->is_parallel = false;
		}
		else if (argv[i] == std::string("-x") && i + 1 < argc)
		{
			cfg->x_threads = std::stoi(argv[++i]);
		}
		else if (argv[i] == std::string("-y") && i + 1 < argc)
		{
			cfg->y_threads = std::stoi(argv[++i]);
		}
		else if (argv[i] == std::string("-o"))
		{
			cfg->results_file_path = argv[++i];
		}
		else if (argv[i] == std::string("-cols") && i + 1 < argc)
		{
			cfg->cols = std::stoi(argv[++i]);
		}
		else if (argv[i] == std::string("-rows") && i + 1 < argc)
		{
			cfg->rows = std::stoi(argv[++i]);
		}
	}
}

void print_help()
{
	std::cout << "Usage: mpirun -np <num_processes> ./warsim [options]" << std::endl
			  << std::endl
			  << "Options:" << std::endl
			  << "-h, --help: Print this help message" << std::endl
			  << "-hc, --help-config: Print the config file help message" << std::endl
			  << "-c, --config <path>: Path to config file" << std::endl
			  << "-x <int>: Number of threads on the x axis" << std::endl
			  << "-y <int>: Number of threads on the y axis" << std::endl
			  << "-g, --graphic: Show graphics" << std::endl
			  << "-G, --no-graphic: Don't show graphics" << std::endl
			  << "-nog: same as -G" << std::endl
			  << "-p, --parallel: Run in parallel" << std::endl
			  << "-P, -s, --serial: Run in serial" << std::endl
			  << "-cols <int>: Number of columns" << std::endl
			  << "-rows <int>: Number of rows" << std::endl
			  << "-o <path>: Path to results file" << std::endl
			  << std::endl
			  << "Example: " << std::endl
			  << "mpirun -np 6 ./warsim -c custom-config.cfg -p -g -x 3 -y 2" << std::endl
			  << std::endl
			  << "If no config file is specified, the default config file will be used" << std::endl
			  << "If no results file path is specified, the results will not be written to a file" << std::endl
			  << std::endl;
}

void print_config_help()
{

	std::cout << "The config file is a JSON file with the following format:" << std::endl
			  << "{" << std::endl
			  << "	\"[config_key]\": <config_value>" << std::endl
			  << "}" << std::endl
			  << std::endl
			  << "The following config keys are supported:" << std::endl
			  << "cols: <int>" << std::endl
			  << "rows: <int>" << std::endl
			  << "rand_seed: <int>" << std::endl
			  << "last_generation: <int>" << std::endl
			  << "show_graphics: <bool>" << std::endl
			  << "is_parallel: <bool>" << std::endl
			  << "x_threads: <int>" << std::endl
			  << "y_threads: <int>" << std::endl
			  << "results_file_path: <string>" << std::endl
			  << "cell_size: <int>" << std::endl
			  << "cell_width: <int>" << std::endl
			  << "cell_height: <int>" << std::endl
			  << "draw_edges: <bool>" << std::endl
			  << "draw_threads_grid: <bool>" << std::endl
			  << "wall_color: [r, g, b], where r,g,b are int between 0-255" << std::endl
			  << "floor_color: [r, g, b], where r,g,b are int between 0-255" << std::endl
			  << "threads_grid_color: [r, g, b], where r,g,b are int between 0-255" << std::endl;
}

inline void exit()
{
	MPI_Abort(MPI_COMM_WORLD, 1);
	exit(1);
}
