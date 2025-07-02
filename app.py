import streamlit as st
import whisper
import tempfile
import os
from pathlib import Path
import subprocess
from typing import Optional, Dict, Any
import time
import zipfile
import io

# Page configuration
st.set_page_config(
    page_title="Auto Subtitle Generator",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

class StreamlitSubtitleGenerator:
    """Streamlit-optimized version of the subtitle generator"""
    
    def __init__(self):
        self.model = None
        self.current_model_size = None
    
    @st.cache_resource
    def load_model(_self, model_size: str):
        """Load Whisper model with caching"""
        if _self.current_model_size != model_size:
            with st.spinner(f'Loading {model_size} model...'):
                _self.model = whisper.load_model(model_size)
                _self.current_model_size = model_size
        return _self.model
    
    def extract_audio_from_video(self, video_path: str, audio_path: str) -> bool:
        """Extract audio from video file"""
        try:
            cmd = [
                "ffmpeg", "-i", video_path,
                "-vn", "-acodec", "pcm_s16le",
                "-ar", "16000", "-ac", "1",
                "-y", audio_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.returncode == 0
        except Exception:
            return False
    
    def generate_subtitles(self, audio_path: str, language: Optional[str], task: str) -> Dict[str, Any]:
        """Generate subtitles from audio"""
        options = {"task": task, "fp16": False}
        if language and language != "auto":
            options["language"] = language
            
        result = self.model.transcribe(audio_path, **options)
        return result
    
    def format_timestamp(self, seconds: float) -> str:
        """Format timestamp for SRT format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"
    
    def create_srt_content(self, result: Dict[str, Any]) -> str:
        """Create SRT file content"""
        srt_content = ""
        for i, segment in enumerate(result["segments"], 1):
            start_time = self.format_timestamp(segment["start"])
            end_time = self.format_timestamp(segment["end"])
            text = segment["text"].strip()
            
            srt_content += f"{i}\n"
            srt_content += f"{start_time} --> {end_time}\n"
            srt_content += f"{text}\n\n"
        
        return srt_content
    
    def create_vtt_content(self, result: Dict[str, Any]) -> str:
        """Create VTT file content"""
        vtt_content = "WEBVTT\n\n"
        for segment in result["segments"]:
            start_time = self.format_timestamp(segment["start"]).replace(',', '.')
            end_time = self.format_timestamp(segment["end"]).replace(',', '.')
            text = segment["text"].strip()
            
            vtt_content += f"{start_time} --> {end_time}\n"
            vtt_content += f"{text}\n\n"
        
        return vtt_content

def main():
    # Initialize generator
    if 'generator' not in st.session_state:
        st.session_state.generator = StreamlitSubtitleGenerator()
    
    # Header
    st.title("🎬 Auto Subtitle Generator")
    st.markdown("### Convert any video or audio file to subtitles using AI")
    st.markdown("---")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model selection
        model_size = st.selectbox(
            "Model Quality",
            options=["tiny", "base", "small", "medium", "large"],
            index=1,  # Default to "base"
            help="""
            - **tiny**: Fastest, good for testing (39MB)
            - **base**: Good balance of speed/accuracy (74MB) ⭐
            - **small**: Better accuracy (244MB)
            - **medium**: High accuracy (769MB)
            - **large**: Best accuracy (1550MB)
            """
        )
        
        # Language selection
        language = st.selectbox(
            "Language",
            options=["auto", "en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh"],
            help="Leave on 'auto' for automatic language detection"
        )
        
        # Task selection
        task = st.selectbox(
            "Task",
            options=["transcribe", "translate"],
            help="""
            - **transcribe**: Keep original language
            - **translate**: Translate to English
            """
        )
        
        # Output format
        output_format = st.selectbox(
            "Output Format",
            options=["srt", "vtt", "both"],
            help="""
            - **SRT**: Standard subtitle format (works with most players)
            - **VTT**: Web video format (for HTML5 players)
            - **Both**: Generate both formats
            """
        )
        
        st.markdown("---")
        st.markdown("### 🎯 Supported Formats")
        st.markdown("""
        **Video**: MP4, AVI, MOV, MKV, WebM, FLV
        **Audio**: MP3, WAV, FLAC, M4A, OGG
        """)
        
        st.markdown("---")
        st.markdown("### 📱 How to use subtitles")
        st.markdown("""
        1. **VLC Player**: Subtitle → Add Subtitle File
        2. **YouTube**: Upload SRT when uploading video
        3. **Most Players**: Same filename as video
        """)
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📁 Upload File")
        uploaded_file = st.file_uploader(
            "Choose a video or audio file",
            type=['mp4', 'avi', 'mov', 'mkv', 'webm', 'flv', 'mp3', 'wav', 'flac', 'm4a', 'ogg'],
            help="Maximum file size: 200MB"
        )
        
        if uploaded_file is not None:
            # Display file info
            file_size = len(uploaded_file.getvalue()) / 1024 / 1024  # MB
            st.success(f"✅ File uploaded: {uploaded_file.name} ({file_size:.1f} MB)")
            
            # Process button
            if st.button("🚀 Generate Subtitles", type="primary", use_container_width=True):
                process_file(uploaded_file, model_size, language, task, output_format)
    
    with col2:
        st.header("ℹ️ Info")
        st.info("""
        **Processing Time Estimates:**
        - Tiny model: ~0.5x real-time
        - Base model: ~1x real-time  
        - Large model: ~3x real-time
        
        **First run**: Downloads model (may take a few minutes)
        **Subsequent runs**: Much faster!
        """)
        
        # Example section
        with st.expander("📋 Sample SRT Format"):
            st.code("""1
00:00:00,000 --> 00:00:03,500
Welcome to this tutorial

2
00:00:03,500 --> 00:00:07,000
Today we'll learn about AI""", language="text")

def process_file(uploaded_file, model_size, language, task, output_format):
    """Process the uploaded file"""
    
    # Create progress bars
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Save uploaded file
        status_text.text("💾 Saving uploaded file...")
        progress_bar.progress(10)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            input_path = tmp_file.name
        
        # Step 2: Load model
        status_text.text(f"🤖 Loading {model_size} model...")
        progress_bar.progress(20)
        
        generator = st.session_state.generator
        model = generator.load_model(model_size)
        
        # Step 3: Extract audio if needed
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv'}
        file_ext = Path(uploaded_file.name).suffix.lower()
        
        if file_ext in video_extensions:
            status_text.text("🎵 Extracting audio from video...")
            progress_bar.progress(40)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as audio_file:
                audio_path = audio_file.name
            
            if not generator.extract_audio_from_video(input_path, audio_path):
                st.error("❌ Failed to extract audio from video. Please check the file format.")
                return
            
            cleanup_audio = True
        else:
            audio_path = input_path
            cleanup_audio = False
        
        # Step 4: Generate subtitles
        status_text.text("🔄 Generating subtitles with AI...")
        progress_bar.progress(60)
        
        start_time = time.time()
        result = generator.generate_subtitles(
            audio_path, 
            language if language != "auto" else None, 
            task
        )
        processing_time = time.time() - start_time
        
        progress_bar.progress(80)
        
        # Step 5: Create subtitle files
        status_text.text("📝 Creating subtitle files...")
        
        files_created = []
        file_stem = Path(uploaded_file.name).stem
        
        if output_format in ["srt", "both"]:
            srt_content = generator.create_srt_content(result)
            files_created.append((f"{file_stem}.srt", srt_content))
        
        if output_format in ["vtt", "both"]:
            vtt_content = generator.create_vtt_content(result)
            files_created.append((f"{file_stem}.vtt", vtt_content))
        
        progress_bar.progress(100)
        status_text.text("✅ Processing complete!")
        
        # Display results
        st.success(f"🎉 Subtitles generated successfully in {processing_time:.1f} seconds!")
        
        # Show statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Segments", len(result["segments"]))
        with col2:
            if result["segments"]:
                duration = result["segments"][-1]["end"]
                st.metric("Duration", f"{duration:.1f}s")
        with col3:
            detected_lang = result.get("language", "unknown")
            st.metric("Language", detected_lang.upper())
        
        # Preview subtitles
        st.header("📖 Subtitle Preview")
        preview_text = result["text"][:500]
        if len(result["text"]) > 500:
            preview_text += "..."
        st.text_area("First 500 characters:", preview_text, height=100)
        
        # Download section
        st.header("💾 Download Files")
        
        if len(files_created) == 1:
            # Single file download
            filename, content = files_created[0]
            st.download_button(
                label=f"📥 Download {filename}",
                data=content,
                file_name=filename,
                mime="text/plain",
                use_container_width=True
            )
        else:
            # Multiple files - create zip
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for filename, content in files_created:
                    zip_file.writestr(filename, content)
            
            st.download_button(
                label="📥 Download All Files (ZIP)",
                data=zip_buffer.getvalue(),
                file_name=f"{file_stem}_subtitles.zip",
                mime="application/zip",
                use_container_width=True
            )
        
        # Individual file downloads
        for filename, content in files_created:
            with st.expander(f"📄 {filename}"):
                st.text_area(f"Content of {filename}:", content, height=200)
                st.download_button(
                    label=f"Download {filename}",
                    data=content,
                    file_name=filename,
                    mime="text/plain",
                    key=f"download_{filename}"
                )
        
        # Cleanup
        os.unlink(input_path)
        if cleanup_audio:
            os.unlink(audio_path)
            
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        progress_bar.progress(0)
        status_text.text("❌ Processing failed")
    
    finally:
        # Clean up progress indicators
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()

if __name__ == "__main__":
    main()