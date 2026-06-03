// record.c
//
// Build:
//   gcc -O2 -Wall -o record record.c -lasound
//
// Run:
//   ./record
//   ./record plughw:1,0
//   ./record hw:2,0 myfile.wav
//
// Behavior:
// - No args: auto-detect first USB capture device, output to recorded.wav
// - 1 arg:   use arg as ALSA device, output to recorded.wav
// - 2 args:  use arg1 as ALSA device, arg2 as output filename
//
// Stop recording with Ctrl+C.

#include <alsa/asoundlib.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define DEFAULT_OUTPUT     "recorded.wav"
#define SAMPLE_RATE        48000
#define CHANNELS           1
#define BITS_PER_SAMPLE    16
#define PERIOD_FRAMES      1024

static volatile sig_atomic_t g_stop = 0;

typedef struct {
    char     riff[4];
    uint32_t overall_size;
    char     wave[4];
    char     fmt_chunk_marker[4];
    uint32_t length_of_fmt;
    uint16_t format_type;
    uint16_t channels;
    uint32_t sample_rate;
    uint32_t byterate;
    uint16_t block_align;
    uint16_t bits_per_sample;
    char     data_chunk_header[4];
    uint32_t data_size;
} wav_header_t;

static void on_sigint(int signo) {
    (void)signo;
    g_stop = 1;
}

static void write_wav_header(FILE *f, uint32_t data_size, unsigned int sample_rate, unsigned int channels, unsigned int bits_per_sample) {
    wav_header_t h;

    memcpy(h.riff, "RIFF", 4);
    h.overall_size = 36 + data_size;
    memcpy(h.wave, "WAVE", 4);
    memcpy(h.fmt_chunk_marker, "fmt ", 4);
    h.length_of_fmt = 16;
    h.format_type = 1; // PCM
    h.channels = (uint16_t)channels;
    h.sample_rate = sample_rate;
    h.bits_per_sample = (uint16_t)bits_per_sample;
    h.byterate = sample_rate * channels * (bits_per_sample / 8);
    h.block_align = channels * (bits_per_sample / 8);
    memcpy(h.data_chunk_header, "data", 4);
    h.data_size = data_size;

    fseek(f, 0, SEEK_SET);
    fwrite(&h, sizeof(h), 1, f);
}

static int strcasestr_bool(const char *haystack, const char *needle) {
    if (!haystack || !needle) return 0;

    size_t nlen = strlen(needle);
    if (nlen == 0) return 1;

    for (; *haystack; haystack++) {
        size_t i;
        for (i = 0; i < nlen; i++) {
            if (!haystack[i]) return 0;
            if (tolower((unsigned char)haystack[i]) != tolower((unsigned char)needle[i])) {
                break;
            }
        }
        if (i == nlen) return 1;
    }
    return 0;
}

static int find_first_usb_capture_device(char *out, size_t out_sz) {
    int card = -1;

    if (!out || out_sz == 0) {
        return -1;
    }

    if (snd_card_next(&card) < 0) {
        return -1;
    }

    while (card >= 0) {
        snd_ctl_t *ctl = NULL;
        char ctl_name[32];
        snd_ctl_card_info_t *card_info;
        int dev = -1;

        snprintf(ctl_name, sizeof(ctl_name), "hw:%d", card);

        if (snd_ctl_open(&ctl, ctl_name, 0) >= 0) {
            snd_ctl_card_info_malloc(&card_info);
            if (snd_ctl_card_info(ctl, card_info) >= 0) {
                const char *card_name = snd_ctl_card_info_get_name(card_info);
                const char *long_name = snd_ctl_card_info_get_longname(card_info);

                int looks_usb = strcasestr_bool(card_name, "usb") ||
                                strcasestr_bool(long_name, "usb");

                if (looks_usb) {
                    if (snd_ctl_pcm_next_device(ctl, &dev) >= 0) {
                        while (dev >= 0) {
                            snd_pcm_info_t *pcminfo;
                            snd_pcm_info_malloc(&pcminfo);
                            snd_pcm_info_set_device(pcminfo, dev);
                            snd_pcm_info_set_subdevice(pcminfo, 0);
                            snd_pcm_info_set_stream(pcminfo, SND_PCM_STREAM_CAPTURE);

                            if (snd_ctl_pcm_info(ctl, pcminfo) >= 0) {
                                snprintf(out, out_sz, "plughw:%d,%d", card, dev);
                                snd_pcm_info_free(pcminfo);
                                snd_ctl_card_info_free(card_info);
                                snd_ctl_close(ctl);
                                return 0;
                            }

                            snd_pcm_info_free(pcminfo);
                            if (snd_ctl_pcm_next_device(ctl, &dev) < 0) {
                                break;
                            }
                        }
                    }
                }
            }
            snd_ctl_card_info_free(card_info);
            snd_ctl_close(ctl);
        }

        if (snd_card_next(&card) < 0) {
            break;
        }
    }

    return -1;
}

int main(int argc, char *argv[]) {
    const char *device = NULL;
    const char *outfile = DEFAULT_OUTPUT;
    char autodetected_device[64];

    snd_pcm_t *pcm = NULL;
    snd_pcm_hw_params_t *hw_params = NULL;
    FILE *f = NULL;
    int16_t *buffer = NULL;

    unsigned int rate = SAMPLE_RATE;
    unsigned int channels = CHANNELS;
    unsigned int bits_per_sample = BITS_PER_SAMPLE;
    snd_pcm_uframes_t frames = PERIOD_FRAMES;
    int dir = 0;
    int err;

    size_t frame_bytes = channels * (bits_per_sample / 8);
    size_t buffer_bytes = frames * frame_bytes;
    uint32_t total_data_bytes = 0;

    if (argc >= 2) {
        device = argv[1];
    } else {
        if (find_first_usb_capture_device(autodetected_device, sizeof(autodetected_device)) != 0) {
            fprintf(stderr, "Could not find a USB capture device.\n");
            fprintf(stderr, "Try specifying one manually, e.g. ./record_usb_default plughw:1,0\n");
            return 1;
        }
        device = autodetected_device;
    }

    if (argc >= 3) {
        outfile = argv[2];
    }

    signal(SIGINT, on_sigint);

    f = fopen(outfile, "wb");
    if (!f) {
        perror("fopen");
        return 1;
    }

    // Placeholder 44-byte WAV header
    {
        uint8_t zero_header[44] = {0};
        if (fwrite(zero_header, 1, sizeof(zero_header), f) != sizeof(zero_header)) {
            perror("fwrite");
            fclose(f);
            return 1;
        }
    }

    if ((err = snd_pcm_open(&pcm, device, SND_PCM_STREAM_CAPTURE, 0)) < 0) {
        fprintf(stderr, "snd_pcm_open(%s) failed: %s\n", device, snd_strerror(err));
        fclose(f);
        return 1;
    }

    snd_pcm_hw_params_malloc(&hw_params);
    snd_pcm_hw_params_any(pcm, hw_params);

    if ((err = snd_pcm_hw_params_set_access(pcm, hw_params, SND_PCM_ACCESS_RW_INTERLEAVED)) < 0) {
        fprintf(stderr, "set_access failed: %s\n", snd_strerror(err));
        goto cleanup;
    }

    if ((err = snd_pcm_hw_params_set_format(pcm, hw_params, SND_PCM_FORMAT_S16_LE)) < 0) {
        fprintf(stderr, "set_format failed: %s\n", snd_strerror(err));
        goto cleanup;
    }

    if ((err = snd_pcm_hw_params_set_channels(pcm, hw_params, channels)) < 0) {
        fprintf(stderr, "set_channels failed: %s\n", snd_strerror(err));
        goto cleanup;
    }

    if ((err = snd_pcm_hw_params_set_rate_near(pcm, hw_params, &rate, &dir)) < 0) {
        fprintf(stderr, "set_rate failed: %s\n", snd_strerror(err));
        goto cleanup;
    }

    if ((err = snd_pcm_hw_params_set_period_size_near(pcm, hw_params, &frames, &dir)) < 0) {
        fprintf(stderr, "set_period_size failed: %s\n", snd_strerror(err));
        goto cleanup;
    }

    if ((err = snd_pcm_hw_params(pcm, hw_params)) < 0) {
        fprintf(stderr, "snd_pcm_hw_params failed: %s\n", snd_strerror(err));
        goto cleanup;
    }

    if (rate != SAMPLE_RATE) {
        fprintf(stderr, "Warning: using %u Hz instead of requested %u Hz\n", rate, SAMPLE_RATE);
    }

    frame_bytes = channels * (bits_per_sample / 8);
    buffer_bytes = frames * frame_bytes;

    buffer = (int16_t *)malloc(buffer_bytes);
    if (!buffer) {
        fprintf(stderr, "malloc failed\n");
        goto cleanup;
    }

    fprintf(stderr, "Recording from %s to %s\n", device, outfile);
    fprintf(stderr, "Format: %u Hz, %u channel(s), %u-bit PCM\n", rate, channels, bits_per_sample);
    fprintf(stderr, "Press Ctrl+C to stop.\n");

    while (!g_stop) {
        snd_pcm_sframes_t n = snd_pcm_readi(pcm, buffer, frames);

        if (n == -EPIPE) {
            fprintf(stderr, "Overrun occurred, recovering...\n");
            snd_pcm_prepare(pcm);
            continue;
        } else if (n < 0) {
            fprintf(stderr, "read failed: %s\n", snd_strerror((int)n));
            break;
        }

        if (n > 0) {
            size_t bytes_to_write = (size_t)n * frame_bytes;
            if (fwrite(buffer, 1, bytes_to_write, f) != bytes_to_write) {
                perror("fwrite");
                break;
            }
            total_data_bytes += (uint32_t)bytes_to_write;
        }
    }

    write_wav_header(f, total_data_bytes, rate, channels, bits_per_sample);
    fprintf(stderr, "\nStopped. Wrote %u bytes of audio data to %s\n", total_data_bytes, outfile);

cleanup:
    if (buffer) free(buffer);
    if (hw_params) snd_pcm_hw_params_free(hw_params);
    if (pcm) snd_pcm_close(pcm);
    if (f) fclose(f);

    return 0;
}
