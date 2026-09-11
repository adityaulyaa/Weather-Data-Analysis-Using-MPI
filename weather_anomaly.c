#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

#define MAX_LINE 1024
#define MAX_RECORDS 5000

typedef struct {
    char date[16];
    double max_temp;
    double rainfall;
} WeatherRecord;

int load_weather_csv(const char *filename, WeatherRecord *dataset) {
    // Baca File CSV
    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("[Error Rank 0] Gagal membuka file '%s'! Pastikan file berada di folder ini.\n", filename);
        fflush(stdout);
        return -1;
    }

    char line[MAX_LINE];
    int count = 0;

    // Abaikan baris header
    if (fgets(line, sizeof(line), file) == NULL) {
        fclose(file);
        return 0;
    }
    // Membaca Setiap Baris
    while (fgets(line, sizeof(line), file) && count < MAX_RECORDS) {
        line[strcspn(line, "\r\n")] = 0;

        char *token;
        char date_str[16] = "";
        double max_temp = -999.0;
        double rainfall = -999.0;
        int col = 0;

        token = strtok(line, ",");
        while (token != NULL) {
            if (col == 0) {
                strncpy(date_str, token, sizeof(date_str) - 1);
            } else if (col == 2) {
                if (strcmp(token, "NA") != 0 && strlen(token) > 0) {
                    max_temp = atof(token);
                }
            } else if (col == 3) {
                if (strcmp(token, "NA") != 0 && strlen(token) > 0) {
                    rainfall = atof(token);
                }
            }
            token = strtok(NULL, ",");
            col++;
        }

        if (max_temp != -999.0) {
            strncpy(dataset[count].date, date_str, sizeof(dataset[count].date) - 1);
            dataset[count].max_temp = max_temp;
            dataset[count].rainfall = (rainfall != -999.0) ? rainfall : 0.0;
            count++;
        }
    }

    fclose(file);
    return count;
}

int main(int argc, char **argv) {
    int rank, size;
    WeatherRecord *full_dataset = NULL;
    int total_records = 0;
    double start_time, end_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    start_time = MPI_Wtime();

    if (rank == 0) {
        printf("[Master] Memulai pembacaan file Weather_Data.csv...\n");
        fflush(stdout);

        full_dataset = (WeatherRecord *)malloc(MAX_RECORDS * sizeof(WeatherRecord));
        total_records = load_weather_csv("Weather_Data.csv", full_dataset);

        if (total_records > 0) {
            printf("[Master] Berhasil membaca %d data cuaca valid.\n", total_records);
            fflush(stdout);
        } else {
            printf("[Master] Warning: Data dibaca 0 baris.\n");
            fflush(stdout);
        }
    }

    MPI_Bcast(&total_records, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (total_records <= 0) {
        if (rank == 0 && full_dataset) free(full_dataset);
        MPI_Finalize();
        return 0;
    }

    int chunk_size = total_records / size;
    int remainder = total_records % size;

    double global_mean = 0.0, global_std = 0.0;
    if (rank == 0) {
        double sum = 0.0;
        for (int i = 0; i < total_records; i++) {
            sum += full_dataset[i].max_temp;
        }
        global_mean = sum / total_records;

        double sq_diff_sum = 0.0;
        for (int i = 0; i < total_records; i++) {
            sq_diff_sum += pow(full_dataset[i].max_temp - global_mean, 2);
        }
        global_std = sqrt(sq_diff_sum / total_records);

        printf("[Master] Statistik Global -> Rata-rata Suhu Max: %.2f°C | Std Deviasi: %.2f°C\n", 
               global_mean, global_std);
        fflush(stdout);
    }

    MPI_Bcast(&global_mean, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&global_std, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int local_count = (rank == size - 1) ? (chunk_size + remainder) : chunk_size;
    double *local_temps = (double *)malloc(local_count * sizeof(double));

    int *send_counts = NULL;
    int *displs = NULL;
    double *all_temps = NULL;

    if (rank == 0) {
        send_counts = (int *)malloc(size * sizeof(int));
        displs = (int *)malloc(size * sizeof(int));
        all_temps = (double *)malloc(total_records * sizeof(double));

        for (int i = 0; i < total_records; i++) {
            all_temps[i] = full_dataset[i].max_temp;
        }

        int offset = 0;
        for (int i = 0; i < size; i++) {
            send_counts[i] = (i == size - 1) ? (chunk_size + remainder) : chunk_size;
            displs[i] = offset;
            offset += send_counts[i];
        }
    }

    MPI_Scatterv(all_temps, send_counts, displs, MPI_DOUBLE,
                 local_temps, local_count, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int local_anomalies = 0;
    for (int i = 0; i < local_count; i++) {
        if (fabs(local_temps[i] - global_mean) > (2.0 * global_std)) {
            local_anomalies++;
        }
    }

    printf(" -> Node Rank %d memproses %d data | Ditemukan %d Anomali Suhu\n", 
           rank, local_count, local_anomalies);
    fflush(stdout);

    int total_anomalies = 0;
    MPI_Reduce(&local_anomalies, &total_anomalies, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    end_time = MPI_Wtime();

    if (rank == 0) {
        printf("\n================ HASIL ANALISIS PARALEL MPI ================\n");
        printf(" Total Data Diproses    : %d baris\n", total_records);
        printf(" Jumlah Node Paralel    : %d proses\n", size);
        printf(" Total Anomali Ditemukan: %d data\n", total_anomalies);
        printf(" Waktu Eksekusi (MPI)   : %f detik\n", end_time - start_time);
        printf("============================================================\n");
        fflush(stdout);

        free(full_dataset);
        free(all_temps);
        free(send_counts);
        free(displs);
    }

    free(local_temps);
    MPI_Finalize();
    return 0;
}