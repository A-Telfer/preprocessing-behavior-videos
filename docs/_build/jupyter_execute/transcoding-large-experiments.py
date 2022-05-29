#!/usr/bin/env python
# coding: utf-8

# # Transcoding: A Practical Example with a Large Experiment 
# Contact: andretelfer@cmail.carleton.ca
# 
# ## Introduction
# ### What is transcoding and why do we do it?
# Transcoding is the process of converting video from one format to another. 
# 
# In behavior, this is useful because we may want to compress videos to use as little disk space as possible or to lower their resolution to speed up later deep learning models.
# 
# ### What's covered here
# Here we explore a large experiment with an inconsistent structure and lots of irrelevant data
# 
# We're going to focus on a few main tasks (and the validation of them)
# 1. Exploring the file systems: Finding all of the relevant videos
# 2. Video details: Getting high level details from them to later verify everything was copied correctly (such as time). We also want to see if the videos themselves are very different and need lots of preprocessing.
# 3. Transcoding the videos into a new folder using `ffmpeg`
# 
# ```{note}
# In this case, we do not resize the videos because they are fairly low quality. However you can easily add option to the ffmpeg command in the transcoding step to this.
# ```
# 
# ### What's not covered here
# This notebook is aimed at researchers, in order to understand what's going on it's helpful to be somewhat familiar with 
# - python
# - bash 
# - ffmpeg
# 
# For questions or suggestions, reach out to me at andretelfer@cmail.carleton.ca

# ## 1. Exploring the file system

# Lets first find which drives are mounted

# In[1]:


ls /media/andre


# I know that the first 3 drives contain experimental data, the last `maternal` drive is where I plan to store the transcoded videos

# In[2]:


from pathlib import Path 

# On my system, this is where the storage devices are mounted
MOUNT_POINT = Path("/media/andre")

# The names of drives the data exists
DRIVES = [
    '11D9-5C57',
    '5161-4A93',
    '9B57-8640'
]

# Where we want to store the videos
OUTPUT_DRIVE = MOUNT_POINT / 'maternal'


# ### Size of Original Datasets

# Lets see how big our original dataset is

# In[3]:


get_ipython().run_cell_magic('time', '', '\ntotal_size = 0\nfor drive in DRIVES:\n    # glob is a useful tool for searching folders, here we tell it to find every file\n    files = (MOUNT_POINT / drive).glob(\'**/*\')\n    \n    # Only keep the files (discard directories)\n    files = list(filter(lambda x: x.is_file(), files)) \n    \n    # Get the size of each file\n    sizes = list(map(lambda x: x.stat().st_size, files)) \n    size = sum(sizes)\n    total_size += size\n    \n    # print the size in gigabytes\n    print(f"Drive {drive} is {size / 1e9:.2f}GB")\n    \n# print the total size\nprint(f"Total: {total_size / 1e9:.2f}GB")')


# ### What types of videos are in the dataset

# Eek, over 2TB of data. But do we really need all of these files? I only want the videos to transcode, and can ignore everything else
# 
# Since videos can have many file extensions, lets print out all of the file extensions so we can identify the video ones.

# In[4]:


total_size = 0

# Use a set instead of a list to ignore duplicates
extensions = set()
for drive in DRIVES:
    files = (MOUNT_POINT / drive).glob('**/*')
    files = list(filter(lambda x: x.is_file(), files))
    
    # Get the extensions
    for filepath in files:
        fileparts = filepath.parts[-1].split('.')
        
        # Some files don't have extensions, ignore those
        if len(fileparts) > 1:
            extensions.add(fileparts[-1])

print("Extensions: ", list(extensions))


# ### Further narrowing down the videos
# 
# Looking through all of the extensions I can see only 2 video related extensions: mp4 and MPG - every other file we can ignore for now
# 
# However we may not want all of the videos, sometimes experimenters will horde discarded videos in folders like "temp". Let's make sure we only get videos that appear meaningful. 
# 
# ... but first lets check how many videos

# In[5]:


for drive in DRIVES:
    videos = (
        list((MOUNT_POINT / drive).glob('**/*.mp4')) + 
        list((MOUNT_POINT / drive).glob('**/*.MPG'))
    )
    
    print(f"Number of videos in {drive}: {len(videos)}")


# Again, eek. But I can't think of a way around seeing them all so lets print them out anyways.

# In[6]:


for drive in DRIVES:
    videos = (
        list((MOUNT_POINT / drive).glob('**/*.mp4')) + 
        list((MOUNT_POINT / drive).glob('**/*.MPG'))
    )
    videos = list(map(str, videos)) # makes things a bit prettier
    print(videos)


# After lots of reading, I can say fairly confidently there are two types of videos we want to ignore
# 1. Ones that begin with a `.` or are in folders that begin with a `.`. These videos are hidden and are probably artifacts from the camera or some other software.
# 2. Videos in folders that begin with $RECYCLE, 
# 
# For now let's say the rest of the videos are useful, we can sort them out later by viewing them. There are too many to look through each one right now.

# In[7]:


def is_visible(filepath):
    for part in filepath.parts:
        if part.startswith('.'):
            return False
        
    return True

def is_not_recycled(filepath):
    for part in filepath.parts:
        if part.startswith('$RECYCLE'):
            return False
        
    return True
    
all_videos = []
for drive in DRIVES:
    videos = (
        list((MOUNT_POINT / drive).glob('**/*.mp4')) + 
        list((MOUNT_POINT / drive).glob('**/*.MPG'))
    )
    
    all_videos_len = len(videos)
    videos = list(filter(is_visible, videos))
    videos = list(filter(is_not_recycled, videos))
    after_filtering_len = len(videos)
    all_videos += videos
    
    print(f"Length of videos before filtering: {all_videos_len:4}, after: {after_filtering_len}")


# ### Visualizing structure of folders
# Sometimes the folders are very disorganized and its hard to get a big picture of what data we have by looking into each folder one at a time

# In[8]:


pip install -q treelib


# In[9]:


import treelib
from tqdm import tqdm

tree = treelib.Tree()
tree.create_node('/', '/')
for video in tqdm(all_videos):
    video = video.relative_to(MOUNT_POINT)
    parts = video.parts[:-1] # don't include filename
    for i in range(1,len(parts)+1):
        uid = '/'.join(parts[:i])
        name = parts[i-1]
        
        if tree.contains(uid):
            continue 
        
        # include parent
        if i > 1:
            parent_uid = '/'.join(parts[:i-1])
            tree.create_node(name, uid, parent=parent_uid)
        else:
            tree.create_node(name, uid, parent='/')
            
tree.show()


# We can save the whole output to a file to view it there

# ## 2. Video Details

# ### Getting Video Metadata
# 
# So we know how many videos are in each folder, and how big the files are. But how long are the videos? Are they all the same size? (this can be very important because many algorithms struggle with inconsistent scales)
# 
# Knowing this metadata is also important because it can help us validate the transcoded videos later.

# In[10]:


import pandas as pd
import cv2
from tqdm import tqdm

metadata = []
for video in tqdm(all_videos):
    cap = cv2.VideoCapture(str(video))
    _metadata = {
        'file': video,
        'filetype': video.parts[-1].split('.')[-1],
        'frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'fps': float(cap.get(cv2.CAP_PROP_FPS)),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'size': video.stat().st_size
    }
    metadata.append(_metadata)
    
metadata_df = pd.DataFrame(metadata)
metadata_df


# ### Total duration

# In[11]:


(metadata_df.frames / metadata_df.fps).sum()


# ... That doesn't look right. What's going on?

# In[12]:


metadata_df.describe()


# Apparently the minimum number of frames for a video is `-3.074457e+15`... clearly there was a problem there. 
# 
# Let's see what videos are causing the problem

# In[13]:


metadata_df.loc[metadata_df.frames < 0]


# Bad videos. Fortunately there's only a few so we can ignore them and deal with them manually later. Hopefully transcoding them will correct it.

# In[14]:


outlier_rows = metadata_df.frames < 0
metadata_df.loc[outlier_rows, 'frames'] = None
metadata_df.loc[outlier_rows]


# Great! we can get the total duration

# In[15]:


(metadata_df.frames / metadata_df.fps).sum()


# In human language...

# In[16]:


total_seconds = (metadata_df.frames / metadata_df.fps).sum()
total_days = total_seconds / 60 / 60 / 24 # 60s->1m, 60m->1h, 24h->1d
print(f"Total days of videos: {total_days:.1f}")


# Lets consider ourselves fortunate we're not going to score this manually. Scoring just a few hours of videos is a slow process already, scoring 151 days of video would probably take an entire PhD

# ### Any other differences in size, etc?

# In[17]:


metadata_df.describe()


# All of the videos have exactly the same height, and pretty much the same fps. There are videos with different widths however. Lets see all of the widths.

# In[18]:


metadata_df.width.unique()


# Only two video widths, 704px and 720px. Lets see if these are from the two different filetypes `mp4` and `MPG`

# In[19]:


metadata_df.loc[metadata_df.width==704].describe()


# In[20]:


metadata_df.loc[metadata_df.width==704].sample(3)


# In[21]:


metadata_df.loc[metadata_df.width==720].describe()


# In[22]:


metadata_df.loc[metadata_df.width==720].sample(3)


# Yep! At a glance it looks like the mp4 files have a width of 704 and all of the MPG files have a width of 720.
# 
# I'm pretty satisfied with understanding the videos at this point. The differences in videos appear minor, likely due to a camera change. We also know that when we finish transcoding we expect our new videos to have a total duration of about 13082190 seconds.
# 
# We also identified a few videos that may be damaged which we can manually check over later.

# ## 3. Transcoding the videos

# In[23]:


print(f"We're not going to reorganize our {len(all_videos)} videos here.")


# Instead we'll just transcode them over in their original structure for now. The transcoded videos will still be smaller, and we can safely manually reorganize them without risking losing anything since we still have the originals.

# ### Creating a bash file for long runs

# There are many ways to transcode the files in python, however when running long scripts I often use bash. The following script generates a bash file that can be run to transcode all of the files.
# 
# The tool we use to actually do the transcoding is called ffmpeg. It's very versatile, and has way too many options to learn all of them, so you can search up what you need when you need them.
# 
# We can do things like resizing the video or changing quality, in the below command I decrease the video quality slightly using `-crf 24`. This should shrinkg their size a lot.
# 
# Example ffmpeg commands 
# 
# ffmpeg commands don't have to be complicated, a simple one would be 
# ```
# ffmpeg -i <your-input-file> <your-output-file>
# ```
# 
# Hardware encoding (Fast but difficult to control quality)
# ```
# ffmpeg -y -hwaccel cuda -hwaccel_output_format cuda -extra_hw_frames 4 -i {input_file} -c:v h264_nvenc {output_file}
# ```

# In[24]:


import re

with open('transcode.sh', 'w') as fp:
    lines = []
    for video in all_videos:
        relative_path = video.relative_to(MOUNT_POINT)
        output_filepath = OUTPUT_DRIVE / relative_path
        
        # For now lets skip existing filepaths
        if output_filepath.exists():
            continue
        
        cmd = (
            "mkdir -p {output_dir} && " # make a new directory if necessary
            "ffmpeg -y -i {input_file} " # the input file and flags
            "-crf 18 -r 30 -an {output_file}\n" # the output file and flags
        ).format(
            output_dir=re.escape(str(output_filepath.parent)), 
            input_file=re.escape(str(video)), 
            output_file=re.escape(str(output_filepath))
        ) 
        
        lines.append(cmd)
        
    fp.writelines(lines)


# This is what the bash file looks like (but a lot more lines)

# In[25]:


get_ipython().system(' head -n 3 transcode.sh')


# I've now started running this bash script. I'll see you in a few days!

# ### Verifying videos
# ... Well, I'm a bit impatient, so I'm not going to wait a few days. Let's check to see how things are going after a few hours.
# 
# We expect some differences
# - One video to not match/be readable as the bash script is still running and transcoding away as I write this.
# - Other videos will be a few frames off, often some frames are dropped during transcoding

# In[26]:


transcoded_videos = (
    list(OUTPUT_DRIVE.glob('**/*.mp4')) + 
    list(OUTPUT_DRIVE.glob('**/*.MPG'))
)

metadata = []
for transcoded_video in tqdm(transcoded_videos):
    relative_path = transcoded_video.relative_to(OUTPUT_DRIVE)
    original_video = MOUNT_POINT / relative_path
    
    original_cap = cv2.VideoCapture(str(original_video))
    transcoded_cap = cv2.VideoCapture(str(transcoded_video))
    
    _metadata = {
        'original_file': original_video,
        'transcoded_file': transcoded_video,
        'original_frames': int(original_cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'transcoded_frames': int(transcoded_cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'original_filesize': original_video.stat().st_size, 
        'transcoded_filesize': transcoded_video.stat().st_size, 
    }
    metadata.append(_metadata)
    
transcoded_metadata_df = pd.DataFrame(metadata)

# ignore the video currently being transcoded (will show as having 0 frames)
transcoded_metadata_df = transcoded_metadata_df.loc[transcoded_metadata_df.transcoded_frames > 0]

transcoded_metadata_df.head(3)


# In[27]:


transcoded_metadata_df.describe()


# #### Quality checks

# ##### Comparing individual frames
# This is useful to make sure that trying to shrink the videos didn't result in visible differences in quality

# In[28]:


import numpy as np
import matplotlib.pyplot as plt

for idx, row in transcoded_metadata_df.sample(5).iterrows():
    frames = min(row.transcoded_frames, row.original_frames)
    frame_idx = np.random.randint(frames)
    
    original_cap = cv2.VideoCapture(str(row.original_file))
    original_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    
    transcoded_cap = cv2.VideoCapture(str(row.transcoded_file))
    transcoded_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    
    plt.figure(figsize=(20, 8))
    gs = plt.GridSpec(1, 2)
    plt.subplot(gs[0])
    plt.title('Original')
    ret, frame = original_cap.read()
    frame = np.flip(frame, axis=2)
    plt.imshow(frame)
    
    plt.subplot(gs[1])
    plt.title('Transcoded')
    ret, frame = transcoded_cap.read()
    frame = np.flip(frame, axis=2)
    plt.imshow(frame)
    
    plt.show()


# There's no obvious difference between videos in terms of quality

# #### Stacking videos
# We can also stack videos to watch for differences side by side
# ```
# ffmpeg -i original.mp4 -i transcoded.mp4 -filter_complex vstack=inputs=2 stacked-view.mp4
# ```

# ### What have we accomplished?
# This time, we're transcoding in order to reduce filesize, with no change in resolution. Let's see how this is working out...

# In[29]:


print(
    (
        "So far we have transcoded {transcoded_count}/{total_count} videos, "
        "compressing them from {original_size:.1f}MB -> {transcoded_size:.1f}MB. "
        "This is {percentage:.1f}% of the original size, "
        "so we estimate it will shrink the total dataset of {total_original_size:.1f}GB -> {total_transcoded_size:.1f}GB."
    ).format(
        total_count=metadata_df.shape[0],
        transcoded_count=transcoded_metadata_df.shape[0],
        original_size=transcoded_metadata_df.original_filesize.sum()/1e6,
        transcoded_size=transcoded_metadata_df.transcoded_filesize.sum()/1e6,
        percentage=transcoded_metadata_df.transcoded_filesize.sum()/transcoded_metadata_df.original_filesize.sum()*100,
        total_original_size=metadata_df['size'].sum()/1e9,
        total_transcoded_size=metadata_df['size'].sum()/transcoded_metadata_df.original_filesize.sum()/1e9*transcoded_metadata_df.transcoded_filesize.sum(),
    )
)

