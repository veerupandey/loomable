# Feature: multimodal-io, Property 9: RunResult Media Aggregation
"""Property 9: RunResult Media Aggregation.

For any RunResult with model-generated media parts in output.parts and
tool-generated media in tool_activity[].result.metadata["media"], the .images,
.audio, and .videos properties SHALL return lists containing all media of the
respective modality, ordered with model media first followed by tool media in
invocation order. When no media is present, each property SHALL return an empty list.

**Validates: Requirements 6.2, 6.3, 6.4, 6.5, 6.6, 6.7**
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from loomable.agent.run import RunResult
from loomable.content.message import AgentOutput
from loomable.content.parts import MediaPart, Modality, Text
from loomable.kernel.models import ToolOutcome, ToolResult
from loomable.media import Audio, File, Image, Video
from loomable.media.types import _MediaBase


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Strategy: generate a random MediaPart with IMAGE modality
image_media_parts = st.builds(
    MediaPart,
    modality=st.just(Modality.IMAGE),
    media_type=st.sampled_from(["image/png", "image/jpeg", "image/gif", "image/webp"]),
    data=st.binary(min_size=1, max_size=64),
)

# Strategy: generate a random MediaPart with AUDIO modality
audio_media_parts = st.builds(
    MediaPart,
    modality=st.just(Modality.AUDIO),
    media_type=st.sampled_from(["audio/wav", "audio/mp3", "audio/ogg", "audio/flac"]),
    data=st.binary(min_size=1, max_size=64),
)

# Strategy: generate a random MediaPart with VIDEO modality
video_media_parts = st.builds(
    MediaPart,
    modality=st.just(Modality.VIDEO),
    media_type=st.sampled_from(["video/mp4", "video/webm", "video/avi"]),
    data=st.binary(min_size=1, max_size=64),
)

# Strategy: any non-text media part (to use in model output)
any_media_part = st.one_of(image_media_parts, audio_media_parts, video_media_parts)

# Strategy: a text MediaPart (always required for valid AgentOutput)
text_media_part = st.builds(
    MediaPart,
    modality=st.just(Modality.TEXT),
    media_type=st.just("text/plain"),
    data=st.binary(min_size=1, max_size=32),
)

# Strategy: generate Image _MediaBase instances for tool metadata
tool_images = st.builds(
    Image,
    content=st.binary(min_size=1, max_size=64),
    mime_type=st.sampled_from(["image/png", "image/jpeg"]),
)

# Strategy: generate Audio _MediaBase instances for tool metadata
tool_audios = st.builds(
    Audio,
    content=st.binary(min_size=1, max_size=64),
    mime_type=st.sampled_from(["audio/wav", "audio/mp3"]),
)

# Strategy: generate Video _MediaBase instances for tool metadata
tool_videos = st.builds(
    Video,
    content=st.binary(min_size=1, max_size=64),
    mime_type=st.sampled_from(["video/mp4", "video/webm"]),
)

# Strategy: generate File _MediaBase instances for tool metadata
tool_files = st.builds(
    File,
    content=st.binary(min_size=1, max_size=64),
    mime_type=st.just("application/octet-stream"),
    filename=st.text(min_size=1, max_size=20),
)


@st.composite
def tool_outcomes_with_media(draw: st.DrawFn) -> list[ToolOutcome]:
    """Generate a list of ToolOutcome objects, each with media metadata."""
    num_outcomes = draw(st.integers(min_value=0, max_value=4))
    outcomes: list[ToolOutcome] = []
    for i in range(num_outcomes):
        # Each outcome can have a mix of media items
        media_items: list[_MediaBase] = draw(
            st.lists(
                st.one_of(tool_images, tool_audios, tool_videos, tool_files),
                min_size=0,
                max_size=3,
            )
        )
        outcome = ToolOutcome(
            call_id=f"call-{i}",
            result=ToolResult(
                content="tool result text",
                metadata={"media": media_items} if media_items else {},
            ),
        )
        outcomes.append(outcome)
    return outcomes


@st.composite
def run_result_with_media(draw: st.DrawFn) -> tuple[
    RunResult,
    list[MediaPart],  # model image parts
    list[MediaPart],  # model audio parts
    list[MediaPart],  # model video parts
    list[Image],  # tool images (in order)
    list[Audio],  # tool audios (in order)
    list[Video],  # tool videos (in order)
    list[File],  # tool files (in order)
]:
    """Generate a RunResult with tracked model and tool media for verification."""
    # Model output parts: always has at least one text part (AgentOutput requires non-empty)
    model_image_parts: list[MediaPart] = draw(
        st.lists(image_media_parts, min_size=0, max_size=3)
    )
    model_audio_parts: list[MediaPart] = draw(
        st.lists(audio_media_parts, min_size=0, max_size=3)
    )
    model_video_parts: list[MediaPart] = draw(
        st.lists(video_media_parts, min_size=0, max_size=3)
    )

    # Combine with a text part (required for non-empty AgentOutput)
    text_part = draw(text_media_part)
    all_output_parts = [text_part] + model_image_parts + model_audio_parts + model_video_parts
    output = AgentOutput(parts=all_output_parts)

    # Tool activity with media metadata
    num_outcomes = draw(st.integers(min_value=0, max_value=4))
    tool_activity: list[ToolOutcome] = []
    expected_tool_images: list[Image] = []
    expected_tool_audios: list[Audio] = []
    expected_tool_videos: list[Video] = []
    expected_tool_files: list[File] = []

    for i in range(num_outcomes):
        media_items: list[_MediaBase] = draw(
            st.lists(
                st.one_of(tool_images, tool_audios, tool_videos, tool_files),
                min_size=0,
                max_size=3,
            )
        )
        outcome = ToolOutcome(
            call_id=f"call-{i}",
            result=ToolResult(
                content="tool output",
                metadata={"media": media_items} if media_items else {},
            ),
        )
        tool_activity.append(outcome)

        # Track expected items by type (in invocation order)
        for item in media_items:
            if isinstance(item, Image):
                expected_tool_images.append(item)
            elif isinstance(item, Audio):
                expected_tool_audios.append(item)
            elif isinstance(item, Video):
                expected_tool_videos.append(item)
            elif isinstance(item, File):
                expected_tool_files.append(item)

    run_result = RunResult(
        output=output,
        session_id="test-session",
        tool_activity=tool_activity,
    )

    return (
        run_result,
        model_image_parts,
        model_audio_parts,
        model_video_parts,
        expected_tool_images,
        expected_tool_audios,
        expected_tool_videos,
        expected_tool_files,
    )


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


class TestRunResultMediaAggregation:
    """Property 9: RunResult media aggregation correctness."""

    @settings(max_examples=100)
    @given(data=run_result_with_media())
    def test_images_model_first_then_tool_in_order(self, data: tuple) -> None:
        """RunResult.images contains model images first, then tool images in invocation order."""
        (
            run_result,
            model_image_parts,
            _model_audio_parts,
            _model_video_parts,
            expected_tool_images,
            _tool_audios,
            _tool_videos,
            _tool_files,
        ) = data

        images = run_result.images

        # Total count: model images + tool images
        assert len(images) == len(model_image_parts) + len(expected_tool_images)

        # First portion: model images (from_media_part wraps)
        model_portion = images[: len(model_image_parts)]
        for img, original_part in zip(model_portion, model_image_parts):
            assert isinstance(img, Image)
            # Verify the wrapped image corresponds to the original model part
            if original_part.uri is not None:
                assert img.url == original_part.uri
            else:
                assert img.content == original_part.data

        # Second portion: tool images (same instances)
        tool_portion = images[len(model_image_parts) :]
        for img, expected in zip(tool_portion, expected_tool_images):
            assert img is expected

    @settings(max_examples=100)
    @given(data=run_result_with_media())
    def test_audio_model_first_then_tool_in_order(self, data: tuple) -> None:
        """RunResult.audio contains model audio first, then tool audio in invocation order."""
        (
            run_result,
            _model_image_parts,
            model_audio_parts,
            _model_video_parts,
            _tool_images,
            expected_tool_audios,
            _tool_videos,
            _tool_files,
        ) = data

        audio = run_result.audio

        # Total count: model audio + tool audio
        assert len(audio) == len(model_audio_parts) + len(expected_tool_audios)

        # First portion: model audio (from_media_part wraps)
        model_portion = audio[: len(model_audio_parts)]
        for aud, original_part in zip(model_portion, model_audio_parts):
            assert isinstance(aud, Audio)
            if original_part.uri is not None:
                assert aud.url == original_part.uri
            else:
                assert aud.content == original_part.data

        # Second portion: tool audio (same instances)
        tool_portion = audio[len(model_audio_parts) :]
        for aud, expected in zip(tool_portion, expected_tool_audios):
            assert aud is expected

    @settings(max_examples=100)
    @given(data=run_result_with_media())
    def test_videos_model_first_then_tool_in_order(self, data: tuple) -> None:
        """RunResult.videos contains model videos first, then tool videos in invocation order."""
        (
            run_result,
            _model_image_parts,
            _model_audio_parts,
            model_video_parts,
            _tool_images,
            _tool_audios,
            expected_tool_videos,
            _tool_files,
        ) = data

        videos = run_result.videos

        # Total count: model videos + tool videos
        assert len(videos) == len(model_video_parts) + len(expected_tool_videos)

        # First portion: model videos (from_media_part wraps)
        model_portion = videos[: len(model_video_parts)]
        for vid, original_part in zip(model_portion, model_video_parts):
            assert isinstance(vid, Video)
            if original_part.uri is not None:
                assert vid.url == original_part.uri
            else:
                assert vid.content == original_part.data

        # Second portion: tool videos (same instances)
        tool_portion = videos[len(model_video_parts) :]
        for vid, expected in zip(tool_portion, expected_tool_videos):
            assert vid is expected

    @settings(max_examples=100)
    @given(data=run_result_with_media())
    def test_images_only_contains_image_modality(self, data: tuple) -> None:
        """.images only returns Image instances (IMAGE modality items)."""
        (run_result, *_rest) = data

        for item in run_result.images:
            assert isinstance(item, Image)

    @settings(max_examples=100)
    @given(data=run_result_with_media())
    def test_audio_only_contains_audio_modality(self, data: tuple) -> None:
        """.audio only returns Audio instances (AUDIO modality items)."""
        (run_result, *_rest) = data

        for item in run_result.audio:
            assert isinstance(item, Audio)

    @settings(max_examples=100)
    @given(data=run_result_with_media())
    def test_videos_only_contains_video_modality(self, data: tuple) -> None:
        """.videos only returns Video instances (VIDEO modality items)."""
        (run_result, *_rest) = data

        for item in run_result.videos:
            assert isinstance(item, Video)

    @settings(max_examples=100)
    @given(data=run_result_with_media())
    def test_files_only_from_tool_metadata(self, data: tuple) -> None:
        """.files returns only File instances from tool metadata."""
        (
            run_result,
            _model_image_parts,
            _model_audio_parts,
            _model_video_parts,
            _tool_images,
            _tool_audios,
            _tool_videos,
            expected_tool_files,
        ) = data

        files = run_result.files
        assert len(files) == len(expected_tool_files)
        for f, expected in zip(files, expected_tool_files):
            assert f is expected
            assert isinstance(f, File)

    @settings(max_examples=100)
    @given(text_data=st.binary(min_size=1, max_size=32))
    def test_empty_media_returns_empty_lists(self, text_data: bytes) -> None:
        """When no media is present, each property returns an empty list (never None)."""
        # Create a RunResult with only text output and no tool media
        output = AgentOutput(
            parts=[MediaPart(modality=Modality.TEXT, media_type="text/plain", data=text_data)]
        )
        run_result = RunResult(
            output=output,
            session_id="empty-session",
            tool_activity=[],
        )

        assert run_result.images == []
        assert run_result.audio == []
        assert run_result.videos == []
        assert run_result.files == []

        # Verify they are lists, not None
        assert run_result.images is not None
        assert run_result.audio is not None
        assert run_result.videos is not None
        assert run_result.files is not None
