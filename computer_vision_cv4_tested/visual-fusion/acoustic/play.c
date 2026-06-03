#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <errno.h>
#include <alsa/asoundlib.h>

typedef struct {
    uint16_t audio_format;
    uint16_t num_channels;
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bits_per_sample;
    uint32_t data_size;
    long data_offset;
} wav_info_t;

static int read_u16_le(FILE *f, uint16_t *out) {
    uint8_t b[2];
    if (fread(b, 1, 2, f) != 2) return -1;
    *out = (uint16_t)(b[0] | (b[1] << 8));
    return 0;
}

static int read_u32_le(FILE *f, uint32_t *out) {
    uint8_t b[4];
    if (fread(b, 1, 4, f) != 4) return -1;
    *out = (uint32_t)(b[0] | (b[1] << 8) | (b[2] << 16) | ((uint32_t)b[3] << 24));
    return 0;
}

static int parse_wav_header(FILE *f, wav_info_t *info) {
    char id[4];
    uint32_t chunk_size;
    char wave_id[4];

    memset(info, 0, sizeof(*info));

    if (fread(id, 1, 4, f) != 4) return -1;
    if (memcmp(id, "RIFF", 4) != 0) {
        fprintf(stderr, "Not a RIFF file\n");
        return -1;
    }

    if (read_u32_le(f, &chunk_size) < 0) return -1;
    (void)chunk_size;

    if (fread(wave_id, 1, 4, f) != 4) return -1;
    if (memcmp(wave_id, "WAVE", 4) != 0) {
        fprintf(stderr, "Not a WAVE file\n");
        return -1;
    }

    int found_fmt = 0;
    int found_data = 0;

    while (!found_fmt || !found_data) {
        char chunk_id[4];
        uint32_t subchunk_size;

        if (fread(chunk_id, 1, 4, f) != 4) {
            fprintf(stderr, "Unexpected end of file while reading chunks\n");
            return -1;
        }

        if (read_u32_le(f, &subchunk_size) < 0) {
            fprintf(stderr, "Failed to read chunk size\n");
            return -1;
        }

        if (memcmp(chunk_id, "fmt ", 4) == 0) {
            if (read_u16_le(f, &info->audio_format) < 0 ||
                read_u16_le(f, &info->num_channels) < 0 ||
                read_u32_le(f, &info->sample_rate) < 0 ||
                read_u32_le(f, &info->byte_rate) < 0 ||
                read_u16_le(f, &info->block_align) < 0 ||
                read_u16_le(f, &info->bits_per_sample) < 0) {
                fprintf(stderr, "Failed to read fmt chunk\n");
                return -1;
            }

            if (subchunk_size > 16) {
                if (fseek(f, subchunk_size - 16, SEEK_CUR) != 0) {
                    fprintf(stderr, "Failed to skip extra fmt data\n");
                    return -1;
                }
            }

            found_fmt = 1;
        } else if (memcmp(chunk_id, "data", 4) == 0) {
            info->data_size = subchunk_size;
            info->data_offset = ftell(f);
            if (info->data_offset < 0) {
                fprintf(stderr, "ftell failed\n");
                return -1;
            }
            if (fseek(f, subchunk_size, SEEK_CUR) != 0) {
                fprintf(stderr, "Failed to skip data chunk\n");
                return -1;
            }
            found_data = 1;
        } else {
            if (fseek(f, subchunk_size, SEEK_CUR) != 0) {
                fprintf(stderr, "Failed to skip chunk\n");
                return -1;
            }
        }

        if (subchunk_size & 1) {
            if (fseek(f, 1, SEEK_CUR) != 0) {
                fprintf(stderr, "Failed to skip padding byte\n");
                return -1;
            }
        }
    }

    if (info->audio_format != 1) {
        fprintf(stderr, "Unsupported WAV format: only PCM is supported (format=%u)\n",
                info->audio_format);
        return -1;
    }

    if (!(info->bits_per_sample == 8 || info->bits_per_sample == 16)) {
        fprintf(stderr, "Unsupported bits per sample: %u\n", info->bits_per_sample);
        return -1;
    }

    if (!(info->num_channels == 1 || info->num_channels == 2)) {
        fprintf(stderr, "Unsupported channel count: %u\n", info->num_channels);
        return -1;
    }

    if (fseek(f, info->data_offset, SEEK_SET) != 0) {
        fprintf(stderr, "Failed to seek to audio data\n");
        return -1;
    }

    return 0;
}

static snd_pcm_format_t alsa_format_from_wav(const wav_info_t *info) {
    if (info->bits_per_sample == 8) {
        return SND_PCM_FORMAT_U8;
    } else if (info->bits_per_sample == 16) {
        return SND_PCM_FORMAT_S16_LE;
    }
    return SND_PCM_FORMAT_UNKNOWN;
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <file.wav>\n", argv[0]);
        return 1;
    }

    const char *filename = argv[1];
    FILE *f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open %s: %s\n", filename, strerror(errno));
        return 1;
    }

    wav_info_t info;
    if (parse_wav_header(f, &info) != 0) {
        fclose(f);
        return 1;
    }

    snd_pcm_format_t pcm_format = alsa_format_from_wav(&info);
    if (pcm_format == SND_PCM_FORMAT_UNKNOWN) {
        fprintf(stderr, "Unsupported ALSA format\n");
        fclose(f);
        return 1;
    }

    printf("Playing: %s\n", filename);
    printf("Sample rate: %u\n", info.sample_rate);
    printf("Channels: %u\n", info.num_channels);
    printf("Bits/sample: %u\n", info.bits_per_sample);
    printf("Data size: %u bytes\n", info.data_size);

    snd_pcm_t *handle = NULL;
    int err = snd_pcm_open(&handle, "default", SND_PCM_STREAM_PLAYBACK, 0);
    if (err < 0) {
        fprintf(stderr, "snd_pcm_open failed: %s\n", snd_strerror(err));
        fclose(f);
        return 1;
    }

    err = snd_pcm_set_params(
        handle,
        pcm_format,
        SND_PCM_ACCESS_RW_INTERLEAVED,
        info.num_channels,
        info.sample_rate,
        1,
        500000
    );
    if (err < 0) {
        fprintf(stderr, "snd_pcm_set_params failed: %s\n", snd_strerror(err));
        snd_pcm_close(handle);
        fclose(f);
        return 1;
    }

    const size_t buffer_frames = 1024;
    const size_t frame_size = info.block_align;
    const size_t buffer_bytes = buffer_frames * frame_size;

    uint8_t *buffer = (uint8_t *)malloc(buffer_bytes);
    if (!buffer) {
        fprintf(stderr, "Out of memory\n");
        snd_pcm_close(handle);
        fclose(f);
        return 1;
    }

    uint32_t bytes_remaining = info.data_size;

    while (bytes_remaining > 0) {
        size_t to_read = buffer_bytes;
        if (bytes_remaining < to_read) {
            to_read = bytes_remaining;
        }

        size_t nread = fread(buffer, 1, to_read, f);
        if (nread == 0) {
            if (ferror(f)) {
                fprintf(stderr, "Error reading audio data\n");
            }
            break;
        }

        size_t frames = nread / frame_size;
        uint8_t *ptr = buffer;

        while (frames > 0) {
            snd_pcm_sframes_t written = snd_pcm_writei(handle, ptr, frames);

            if (written == -EPIPE) {
                fprintf(stderr, "Underrun occurred\n");
                snd_pcm_prepare(handle);
                continue;
            } else if (written < 0) {
                fprintf(stderr, "snd_pcm_writei failed: %s\n", snd_strerror(written));
                free(buffer);
                snd_pcm_close(handle);
                fclose(f);
                return 1;
            }

            ptr += written * frame_size;
            frames -= written;
        }

        bytes_remaining -= (uint32_t)nread;
    }

    snd_pcm_drain(handle);
    snd_pcm_close(handle);
    free(buffer);
    fclose(f);

    return 0;
}
