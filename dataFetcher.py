import os
import time
import yt_dlp
import requests
import subprocess
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import googleapiclient.discovery
from argparse import ArgumentParser
from multiprocessing import Pool
import math

load_dotenv("keys.env")

# get song data from the .csv file
# connect to yt => get relevant search results
# download the song using yt-dlp
# check if the song fingerprint matches the acoustID

apiServiceName = "youtube"
apiVersion     = "v3"
acoustidURL    = "https://api.acoustid.org/v2/lookup"
THRESHOLD      = 0.7

songsDir       = Path("data/songs/")
fpDir          = Path("data/fingerprints/")
sourcesFile    = songsDir / "sources.csv"
allSourcesFile = songsDir / "allSources.csv"

# if the dirs don't exist yet, create them
songsDir.mkdir(parents = True, exist_ok = True)
fpDir.mkdir(parents = True, exist_ok = True)

metadataFile   = Path("data/metadata.csv")
# metadata for songs (artist, title, album, acoustID, etc.)
metadataDF     = pd.read_csv(metadataFile, index_col = 0)

# yt-dlp
ydl_opts = {
    # be flexible: try bestaudio, then any best format as fallback
    'format': 'bestaudio/best/bestvideo+bestaudio/best',
    'outtmpl': str(songsDir / "postprocessed"), # no extension since the postprocessor will add its own
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'mp3',
        'preferredquality': '0',
    }],
    'verbose': False,
    'quiet': True, # suppress info messages
    'no_warnings': True, # suppress warnings
    'socket_timeout': 60,
    'cookiesfrombrowser': ('firefox',),
    'ignoreerrors': True,
}
youtubeDL = yt_dlp.YoutubeDL(ydl_opts)
dfPlaceholder = songsDir / "postprocessed.mp3"

# inclusive end index
# needed as a separate function as it makes more sense to ping youtube separately (fast returns)
# yet downloading takes forever, so i could just run it later
def fetchYoutubeLinks(startAt: int = 1, maxIndex: int = 5, save: bool = True, results: int = 5) -> pd.DataFrame:
    ids      = {}
    youtube = googleapiclient.discovery.build(
            apiServiceName, 
            apiVersion, 
            developerKey = os.environ.get("GOOGLE_API_KEY_2")
        )
    
    for i in range(startAt, maxIndex + 1):
        request = youtube.search().list(
            part = "snippet",
            maxResults = results,
            q = f"{metadataDF.loc[i].artist} - {metadataDF.loc[i].title} in album {metadataDF.loc[i].album}",
            type = "video"
        )
        response = request.execute()
        if len(response['items']) == 0 or not response['items'][0]['id'].get('videoId'):
            print(i)
            continue

        ids[i] = [item['id'].get('videoId') for item in response['items']][:results]
        time.sleep(0.5)

    df_result = pd.DataFrame.from_dict(ids, orient = 'index')
    # Name columns based on the actual number of columns returned,
    # not on the requested maxResults (API may return fewer/more).
    df_result.columns = [f'result_{j+1}' for j in range(df_result.shape[1])]

    if save:
        df_result.to_csv(allSourcesFile, index_label = 'index')

    print(f"Youtube links fetched for songs {startAt} to {maxIndex}.")
    return df_result

# inclusive end index
def fetchSongs(df: pd.DataFrame, startAt: int = 1, maxIndex: int = 5, results: int = 5, force: bool = False, process_id: int = 0):
    """
    Download candidate songs from YouTube based on the IDs in `df`.
    Does NOT perform any acoustID matching or cleanup.
    """
    # Create yt-dlp instance for this worker (needed for multiprocessing)
    # Use unique placeholder per process to avoid race conditions
    process_opts = ydl_opts.copy()
    process_opts['outtmpl'] = str(songsDir / f"postprocessed_{process_id}")
    ydl = yt_dlp.YoutubeDL(process_opts)
    placeholder = songsDir / f"postprocessed_{process_id}.mp3"
    
    for i in range(startAt, maxIndex + 1):
        # Some indices might not have any YouTube candidates (missing rows in df).
        # In that case, just skip them instead of raising a KeyError.
        if i not in df.index:
            print(f"No YouTube candidates found for song {i}, skipping download.")
            continue
        target = songsDir / f"{i}.mp3"
        if target.exists() and not force:
            print(f"Song {i} already exists, skipping download.")
            continue
        os.remove(target) if target.exists() else None

        N = min(df.loc[i].dropna().shape[0], results)
        for j in range(N):
            id = df.loc[i].dropna().values[j]
            target = songsDir / f"{i}_{j}.mp3"
            if target.exists() and not force:
                print(f"Song {i}_{j} already exists, skipping download.")
                continue
            os.remove(target) if target.exists() else None

            youtubeURL = f"https://www.youtube.com/watch?v={id}"
            meta = ydl.extract_info(youtubeURL, download = False)

            if (meta is None) or (meta.get('duration') is None) or (meta.get('duration') > 1200) or (meta.get('filesize') is not None and meta.get('filesize') > 512 * 1024 * 1024):  # pyright: ignore[reportOptionalMemberAccess, reportOptionalOperand]
                print(f"Video is too big or long (duration = {meta.get('duration')}); probably a collection of songs, so skipping it.")
                continue
            try:
                ydl.download([youtubeURL])
                Path.rename(placeholder, songsDir / f"{i}_{j}.mp3")
            except Exception as e:
                print(f"Failed to download {youtubeURL}: {e}")
                continue
            # time.sleep(2)  # Add delay between downloads to avoid rate limiting


def getBestFit(df: pd.DataFrame, startAt: int = 1, maxIndex: int = 5, results: int = 5, force: bool = False):
    """
    For each song index, run fpcalc + AcoustID lookup on all downloaded
    candidates and keep only the best-matching one.
    Returns dict of best fits: {index: [link, score]}
    """
    bestFit: dict[int, list] = {}
    for i in range(startAt, maxIndex + 1):
        # If this index has no candidates in df, there is nothing to match.
        if i not in df.index:
            print(f"No YouTube candidates found for song {i}, skipping AcoustID matching.")
            continue

        N = min(df.loc[i].dropna().shape[0], results)
        target   = songsDir / f"{i}.mp3"
        if target.exists() and not force:
            print(f"Song {i} is already the best fit.")
            continue
        os.remove(target) if target.exists() else None
        score    = 0.0
        bestID   = -1
        # acoustID to match (from metadata, not from YouTube ID DataFrame)
        acoustID = metadataDF.loc[i, "acoustID"]

        for j in range(N):
            song = songsDir / f"{i}_{j}.mp3"
            if not song.exists():
                continue
            text = fpDir / f"{i}_{j}.txt"
            # the line below executes the fpcalc command in the shell
            subprocess.run(f"fpcalc {song} > {text}", shell = True, check = True)

            with open(text, "r") as f:
                lines = f.readlines()
                duration = int(lines[0].strip().split('=')[1])
                fingerprint = lines[1].strip().split('=')[1]
            
            response = requests.get(acoustidURL, params = {
                "client": os.getenv("ACOUSTID_API_KEY"),
                "format": "json",
                "duration": duration,
                "fingerprint": fingerprint
            })
            time.sleep(1) # to avoid rate limiting
            if response.status_code != 200:
                print(f"AcoustID API request failed for file {i}_{j}.mp3! Exit code {response.status_code}.\n")
            elif len(response.json()['results']) == 0:
                print(f"No AcoustID match found for file {i}_{j}.mp3. \n")

            for d in response.json()['results']:
                if acoustID == d['id'] and d['score'] > score:
                    score  = d['score']
                    bestID = j
            
        if bestID != -1 and score >= THRESHOLD:
            print(f"Found a match with score {score} under the ID {bestID}.")
            link = f"https://www.youtube.com/watch?v={df.loc[i].dropna().values[bestID]}"
            bestFit[i] = [link, score]
            # rename the song to the target file
            os.rename(songsDir / f"{i}_{bestID}.mp3", target)
        elif (bestID == -1) or (score < THRESHOLD):
            print(f"No good match found for song {i}! Best score was {score}. \n")

        for j in range(N):
            song = songsDir / f"{i}_{j}.mp3"
            text = fpDir / f"{i}_{j}.txt"
            if song.exists():
                os.remove(song)
            if text.exists():
                os.remove(text)
    
    return bestFit


def process_chunk(args_tuple):
    """
    Worker function for parallel processing.
    Processes a chunk: fetches songs and gets best fit.
    Returns the bestFit dict for merging.
    """
    df_chunk, metadataDF, chunk_start, chunk_end, results, force, chunk_id = args_tuple
    # Fetch songs for this chunk
    fetchSongs(df_chunk, startAt = chunk_start, maxIndex = chunk_end, results = results, force = force, process_id = chunk_id)
    # Get best fit for this chunk
    return getBestFit(df_chunk, startAt = chunk_start, maxIndex = chunk_end, results = results, force = force)
    
if __name__ == "__main__":
    # handle input arguments
    parser = ArgumentParser()
    parser.add_argument("-n", "--numSongs", type = int, default = 5)
    parser.add_argument("-s", "--startAt", type = int, default = 1)
    parser.add_argument("-e", "--endAt", type = int, default = 5)
    parser.add_argument("-c", "--cores", type = int, default = 4)
    parser.add_argument("-yt", "--fetchYoutubeLinks", action = "store_true", default = False)
    parser.add_argument("--save", action = "store_true", default = True)
    parser.add_argument("-f", "--force", action = "store_true", default = False)
    args = parser.parse_args()

    maxResults = args.numSongs
    maxIndex   = metadataDF.shape[0] if args.endAt == 0 else min(args.endAt, metadataDF.shape[0])
    if args.fetchYoutubeLinks or not allSourcesFile.exists():
        sourcesDF = fetchYoutubeLinks(startAt = args.startAt, maxIndex = maxIndex, results = maxResults, save = args.save)
    else:
        sourcesDF = pd.read_csv(allSourcesFile, index_col = 0)
    
    # Split range into chunks for parallel processing
    num_cores = args.cores
    total_songs = maxIndex - args.startAt + 1
    chunk_size = math.ceil(total_songs / num_cores)
    
    # Create chunks: [(start1, end1), (start2, end2), ...]
    chunks = []
    for i in range(num_cores):
        chunk_start = args.startAt + i * chunk_size
        chunk_end = min(chunk_start + chunk_size - 1, maxIndex)
        if chunk_start > maxIndex:
            break
        chunks.append((sourcesDF, metadataDF, chunk_start, chunk_end, maxResults, args.force, i))
    
    # Process chunks in parallel
    print(f"Processing {len(chunks)} chunks across {num_cores} cores...")
    with Pool(processes=num_cores) as pool:
        results = pool.map(process_chunk, chunks)
    
    # Merge all bestFit results
    all_bestFit = {}
    for result in results:
        all_bestFit.update(result)
    
    # Save merged results
    if not all_bestFit:
        print("No best fits found.")
    else:
        bestFitDF = pd.DataFrame.from_dict(all_bestFit, orient='index')
        bestFitDF.columns = ['link', 'score']
        
        if sourcesFile.exists():
            existingFit = pd.read_csv(sourcesFile, index_col='index')
            bestFitDF = bestFitDF.combine_first(existingFit)
        
        bestFitDF.to_csv(sourcesFile, index_label='index')
        print(f"Saved {len(all_bestFit)} best fits to {sourcesFile}")