#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <keyutils.h>
#include <unistd.h>
#include <time.h>

/*
 * keyctl normal-control workload: churns kernel key objects so the trace shows
 * sustained kmalloc activity in the key buckets.
 *
 * Unbounded, this floods the trace ring buffer: a 2s run issues ~100k ops and
 * produces ~550k events, which overruns the host buffer and invalidates the
 * run. A total-op cap keeps the trace well under the buffer size.
 */

#define NUM_KEYS 5000
#define PAYLOAD_SIZE 256
#define MAX_OPS 45000

int main(int argc, char *argv[]) {
    int duration = 30;
    if (argc > 1) duration = atoi(argv[1]);

    char desc[64], payload[PAYLOAD_SIZE];
    memset(payload, 'B', PAYLOAD_SIZE);

    time_t start = time(NULL);
    int total_ops = 0;

    while (time(NULL) - start < duration && total_ops < MAX_OPS) {
        for (int i = 0; i < NUM_KEYS; i++) {
            snprintf(desc, 64, "wl_key_%d_%ld", i, time(NULL) % 10000);
            key_serial_t key = add_key("user", desc, payload, PAYLOAD_SIZE,
                                       KEY_SPEC_PROCESS_KEYRING);
            total_ops++;
            if (key >= 0 && i % 2 == 0) {
                keyctl(KEYCTL_REVOKE, key);
                total_ops++;
            }
        }
        keyctl(KEYCTL_CLEAR, KEY_SPEC_PROCESS_KEYRING);
        usleep(3000);
    }

    fprintf(stderr, "keyctl workload done: %d ops in %lds\n",
            total_ops, (long)(time(NULL) - start));
    return 0;
}
