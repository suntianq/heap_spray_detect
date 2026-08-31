#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

/*
 * mem_pressure normal-control workload: allocates and frees buffers across a
 * range of sizes (slab kmalloc-64..-4096, page-allocator and mmap-backed
 * anon pages at 64KB/1MB), touching every page so page faults actually allocate.
 * This is a HARD NEGATIVE for the discriminator: sustained high allocation
 * rate and page-fault churn with no spray of the target slab. A total-op cap
 * keeps the run under the host ring buffer (same pattern as the others).
 */

#define MAX_ITERS 3000

static long now_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000L + tv.tv_usec / 1000L;
}

int main(int argc, char *argv[]) {
    int duration = 30;
    if (argc > 1) duration = atoi(argv[1]);
    if (duration < 1) duration = 1;

    long start = now_ms();
    int iters = 0;
    long prev = start;

    while (now_ms() - start < duration * 1000L && iters < MAX_ITERS) {
        /* burst: churn several sizes so multiple buckets show up. */
        for (int i = 0; i < 5 && iters < MAX_ITERS; i++) {
            size_t sz;
            switch (i) {
                case 0: sz = 64; break;          /* kmalloc-64 */
                case 1: sz = 512; break;         /* kmalloc-512 */
                case 2: sz = 4096; break;        /* kmalloc-4k / order-0 pages */
                case 3: sz = 65536; break;       /* order-4 pages */
                default: sz = 1048576; break;    /* mmap anon -> vm_area_struct + page churn */
            }
            void *p = malloc(sz);
            if (p) {
                memset(p, 'M', sz);              /* touch every page -> real alloc */
                free(p);
            }
            iters++;
        }
        /* keep ~5 bursts/sec: no stall on the trace buffer. */
        long elapsed = now_ms() - prev;
        if (elapsed < 200) usleep((200 - elapsed) * 1000);
        prev = now_ms();
    }

    fprintf(stderr, "mem workload done: %d iters in %ldms\n", iters, now_ms() - start);
    return 0;
}
