import cv2
from PIL import Image
import os
import numpy as np
import yt_dlp
import sys


class VideoProcessor:
    # Configuration variables as class variables
    SLIDE_DIR = 'slides'
    MAX_WHITE_PERCENTAGE = 98
    THRESHOLD_DIFF = 2200
    WHITE_THRESHOLD = 100
    TEMP_VIDEO_PATH = 'video_temp.mp4'

    def __init__(self, youtube_url="https://www.youtube.com/watch?v=hxDOyv-Z8_k", resolution='360p'):
        """
        Initializes the VideoProcessor with the YouTube URL and resolution.

        Args:
            youtube_url (str): The URL of the YouTube video to download.
            resolution (str): The desired resolution for the video download.
        """
        self.youtube_url = youtube_url
        self.resolution = resolution
        self.video_path = None

    @classmethod
    def create_directory(cls, directory):
        """
        Creates a directory if it doesn't already exist.

        Args:
            directory (str): The path of the directory to create.
        """
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except Exception as e:
            print(f"Error creating directory {directory}: {e}")

    @classmethod
    def clear_directory(cls, directory):
        """
        Clears all files in the specified directory.

        Args:
            directory (str): The path of the directory to clear.
        """
        try:
            for file in os.listdir(directory):
                file_path = os.path.join(directory, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        except Exception as e:
            print(f"Error clearing directory {directory}: {e}")

    @classmethod
    def remove_temp_video(cls):
        """
        Removes the temporary video file if it exists.
        """
        try:
            if os.path.exists(cls.TEMP_VIDEO_PATH):
                os.remove(cls.TEMP_VIDEO_PATH)
                print("Temporary video file removed.")
        except Exception as e:
            print(f"Error removing temporary video file: {e}")

    def download_video(self):
        """
        Downloads a video from YouTube using the specified resolution.

        Returns:
            str: The path to the downloaded video file.
        """
        self.remove_temp_video()  # Ensure no old video file is left

        try:
            ydl_opts = {
                'format': f'bestvideo[height<={self.resolution[:-1]}]/bestaudio/best',
                'outtmpl': VideoProcessor.TEMP_VIDEO_PATH,
                'noplaylist': True,
                'quiet': True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([self.youtube_url])
                print("Video downloaded successfully.")
            self.video_path = VideoProcessor.TEMP_VIDEO_PATH
        except Exception as e:
            print(f"Error downloading video from {self.youtube_url}: {e}")

    @classmethod
    def proportion_of_almost_white_pixels(cls, frame, threshold=200):
        """
        Calculates the proportion of white or almost white pixels in the image.

        Args:
            frame (numpy.ndarray): The image frame to analyze.
            threshold (int): The threshold value for determining "almost white" pixels.

        Returns:
            float: The proportion of almost white pixels in the image.
        """
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower_white = np.array([0, 0, 255 - threshold])
            upper_white = np.array([180, threshold, 255])
            mask = cv2.inRange(hsv, lower_white, upper_white)
            num_almost_white_pixels = np.sum(mask > 0)
            proportion = num_almost_white_pixels / (frame.shape[0] * frame.shape[1])
            return proportion
        except Exception as e:
            print(f"Error calculating proportion of almost white pixels: {e}")
            return 0

    @classmethod
    def is_valid_proportion_of_almost_white_pixels(cls, frame, threshold=200, max_proportion=0.3):
        """
        Checks if the proportion of white or almost white pixels in the image is below the maximum allowed proportion.

        Args:
            frame (numpy.ndarray): The image frame to analyze.
            threshold (int): The threshold value for determining "almost white" pixels.
            max_proportion (float): The maximum allowed proportion of almost white pixels.

        Returns:
            bool: True if the proportion is within the allowed limit, False otherwise.
        """
        try:
            proportion = cls.proportion_of_almost_white_pixels(frame, threshold)
            return proportion <= max_proportion
        except Exception as e:
            print(f"Error checking valid proportion of almost white pixels: {e}")
            return False

    @classmethod
    def has_significant_frame_difference(cls, previous_frame, current_frame, threshold):
        """
        Determines if there is a significant difference between the previous and current frame.

        Args:
            previous_frame (numpy.ndarray): The previous image frame.
            current_frame (numpy.ndarray): The current image frame.
            threshold (int): The threshold value for determining a significant difference.

        Returns:
            bool: True if the difference between frames exceeds the threshold, False otherwise.
        """
        try:
            diff = cv2.absdiff(previous_frame, current_frame)
            gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
            non_zero_count = cv2.countNonZero(thresh)
            return non_zero_count > threshold
        except Exception as e:
            print(f"Error checking significant frame difference: {e}")
            return False

    def extract_slides(self, threshold=50000):
        """
        Extracts slides from a video based on frame differences and saves them as images.

        Args:
            threshold (int): The threshold value for determining a significant frame difference.

        Returns:
            list: A list of paths to the extracted slide images.
        """
        slide_paths = []
        try:
            cap = cv2.VideoCapture(self.video_path)
            previous_frame = None
            frame_buffer = []
            slide_num = 0

            self.create_directory(VideoProcessor.SLIDE_DIR)
            self.clear_directory(VideoProcessor.SLIDE_DIR)

            while cap.isOpened():
                ret, current_frame = cap.read()
                if not ret:
                    break

                if previous_frame is not None:
                    if self.has_significant_frame_difference(previous_frame, current_frame, threshold):
                        if frame_buffer:
                            frame_with_lowest_proportion = min(
                                frame_buffer,
                                key=lambda f: self.proportion_of_almost_white_pixels(f, VideoProcessor.WHITE_THRESHOLD)
                            )

                            slide_num += 1
                            slide_path = os.path.join(VideoProcessor.SLIDE_DIR, f'slide_{slide_num:03d}.png')
                            cv2.imwrite(slide_path, frame_with_lowest_proportion)

                            slide_paths.append(slide_path)
                            frame_buffer.clear()
                            frame_buffer.append(current_frame)

                if previous_frame is None or self.is_valid_proportion_of_almost_white_pixels(current_frame, VideoProcessor.WHITE_THRESHOLD, VideoProcessor.MAX_WHITE_PERCENTAGE / 100):
                    frame_buffer.append(current_frame)

                previous_frame = current_frame

            if frame_buffer:
                frame_with_lowest_proportion = min(
                    frame_buffer,
                    key=lambda f: self.proportion_of_almost_white_pixels(f, VideoProcessor.WHITE_THRESHOLD)
                )
                slide_num += 1
                slide_path = os.path.join(VideoProcessor.SLIDE_DIR, f'slide_{slide_num:03d}.png')
                cv2.imwrite(slide_path, frame_with_lowest_proportion)
                slide_paths.append(slide_path)

            cap.release()
            print(f'{slide_num} slides extracted.')
        except Exception as e:
            print(f"Error extracting slides: {e}")

        return slide_paths

    @classmethod
    def create_pdf(cls, slide_paths, output_pdf):
        """
        Creates a PDF from a list of slide images.

        Args:
            slide_paths (list): A list of paths to slide images.
            output_pdf (str): The path where the output PDF should be saved.
        """
        try:
            images = [Image.open(slide).convert('RGB') for slide in slide_paths]
            if images:
                images[0].save(output_pdf, save_all=True, append_images=images[1:])
                print(f'PDF created: {output_pdf}')
            else:
                print('No images were processed for the PDF.')
        except Exception as e:
            print(f"Error creating PDF: {e}")

    def process_video(self):
        """
        Main function to process the video, extract slides, and create a PDF.
        """
        try:
            self.download_video()
            slide_paths = self.extract_slides(threshold=VideoProcessor.THRESHOLD_DIFF)
            self.create_pdf(slide_paths, 'slides_output.pdf')
            self.remove_temp_video()  # Cleanup temporary files
        except Exception as e:
            print(f"Error processing video: {e}")

def main():
    """
    Entry point of the script when run directly.
    """
    try:
        if len(sys.argv) > 1:
            youtube_url = sys.argv[1]
            resolution = sys.argv[2] if len(sys.argv) > 2 else '360p'
            processor = VideoProcessor(youtube_url, resolution)
            processor.process_video()
        else:
            print("Usage: python script_name.py <YouTube Video URL> [resolution - optional]")
    except Exception as e:
        print(f"Error in main function: {e}")

if __name__ == "__main__":
    main()
