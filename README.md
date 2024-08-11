# Video Processor

This project was developed to transform a YouTube video lesson with PDF presentations into a PDF with the teacher's notes, avoiding full viewing of the video.

## Features

- **Download Video**: Downloads the YouTube video in a specified resolution.
- **Extract Slides**: Separates the frames into groups, where each frame in a group has minimal difference from the previous frame, and is only included in the group if it does not have a significant percentage of pixels close to white. The separation of groups occurs when the current frame captured shows a significant difference from the previous frame. At that point, the frame with the smallest percentage of white is saved, and the current frame is added to the next group, repeating the process.
- **Create PDF**: Compiles the extracted slides into a PDF file.

## Installation

Make sure you have Python installed. Then, install the necessary dependencies using pip:

```bash
pip install opencv-python pillow numpy yt_dlp
```

# Usage and Example

## Usage

To run the script, use the following command:

```bash
python script_name.py <YouTube Video URL> [resolution - optional]
```

- **`<YouTube Video URL>`**: URL of the YouTube video you want to process.
- **`[resolution - optional]`**: Desired resolution for the video (e.g., '360p'). If not provided, the default resolution is '360p'.


## Example Usage

Here is an example of how to use the script:

```bash
python script_name.py https://www.youtube.com/watch?v=hxDOyv-Z8_k 360p
```

or

```bash
python script_name.py https://www.youtube.com/watch?v=hxDOyv-Z8_k
```

Feel free to replace `script_name.py` with the actual name of your script file.

In this examples:

- The script will download the video from the provided YouTube URL.
- It will process the video at a resolution of '360p'.
- Extracted slides will be saved, and a PDF will be created from these slides.

[Link to some video lessons](https://profmat-sbm.org.br/ma-14/)
