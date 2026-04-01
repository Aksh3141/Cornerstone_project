import os
import subprocess
import argparse

def extract_smart_frames(video_path, output_folder, max_frames=4, threshold=0.1):
    """
    Extracts up to 'max_frames' and returns (path, timestamp) tuples.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for f in os.listdir(output_folder):
        if f.endswith(".jpg"): os.remove(os.path.join(output_folder, f))
            
    print(f"  [FFMPEG] Extracting scenes and timestamps from {os.path.basename(video_path)}...")
    
    # Use showinfo to see exact PTS for each selected frame
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", f"select='gt(scene,{threshold})',scale=1280:-1,showinfo",
        "-vsync", "vfr",
        os.path.join(output_folder, "frame_%04d.jpg")
    ]
    
    try:
        # We capture stderr because showinfo logs there
        process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        stderr_output = process.stderr.decode()
        
        # Parse showinfo output for 'pts_time:X.XXX'
        import re
        pts_matches = re.findall(r"pts_time:(\d+\.?\d*)", stderr_output)
        timestamps = [float(ts) for ts in pts_matches]

        frames = sorted([f for f in os.listdir(output_folder) if f.endswith(".jpg")])
        
        # Correlate frames with timestamps
        frame_data = []
        for i in range(min(len(frames), len(timestamps))):
            frame_data.append((os.path.join(output_folder, frames[i]), timestamps[i]))

        # Sub-sample if too many
        if len(frame_data) > max_frames:
            print(f"    [INFO] Sub-sampling {len(frame_data)} scenes to {max_frames}...")
            indices = [int(i * (len(frame_data) - 1) / (max_frames - 1)) for i in range(max_frames)]
            frame_data = [frame_data[i] for i in indices]
            
            # Clean folders of unselected
            selected_paths = [fd[0] for fd in frame_data]
            for f in frames:
                path = os.path.join(output_folder, f)
                if path not in selected_paths:
                    os.remove(path)

        print(f"  [OK] Extracted {len(frame_data)} timestamped frames.")
        return frame_data
    except subprocess.CalledProcessError as e:
        print(f"  [ERROR] FFMPEG failed: {e.stderr.decode()[:200]}")
        return []

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart frame extraction for OCR.")
    parser.add_argument("--video", required=True, help="Path to input video.")
    parser.add_argument("--output", default="./frames", help="Output directory.")
    parser.add_argument("--max_frames", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.1)
    
    args = parser.parse_args()
    extract_smart_frames(args.video, args.output, args.max_frames, args.threshold)
