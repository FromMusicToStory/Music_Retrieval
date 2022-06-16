import os
import csv
import tarfile
import glob


def unpack_jamendo(input_dir, output_dir):
    ids = {}
    with open('../mtg-jamendo-dataset/data/download/autotagging_moodtheme_audio_gids.txt', 'r') as f:
        for line in f:
            id, filename = line.split(('   '))[:2]
            ids[filename] = id

    file_sha256_tracks = '../mtg-jamendo-dataset/data/download/autotagging_moodtheme_audio_sha256_tars.txt'
    with open(file_sha256_tracks) as f:
        sha256_tracks = dict([(row[1], row[0]) for row in csv.reader(f, delimiter=' ')])

    tracks_checked = []
    for filename in ids:
        if os.path.exists(os.path.join(output_dir, filename.split('-')[1].split('.')[0])):
            print('Skipping %s (file already exists)' % filename.split('-')[1].split('.')[0])

        else:
            print('Unpacking tar archives')
            input = os.path.join(input_dir, filename)
            print('Unpacking', input)
            tar = tarfile.open(input)
            tracks = tar.getnames()[1:]  # The first element is folder name.
            tar.extractall(path=output_dir)
            tar.close()

            tracks_checked += tracks

    # Check if any tracks are missing in the unpacked archives.
    if set(tracks_checked) != set(sha256_tracks.keys()):
        raise Exception('Unpacked data contains tracks not present in the checksum files')

    print('Unpacking complete')


if __name__ == "__main__":
    input_dir = "../mtg-jamendo-dataset/data/download/"
    output_dir = "mtg-jamendo-dataset/"
    unpack_jamendo(input_dir, output_dir)