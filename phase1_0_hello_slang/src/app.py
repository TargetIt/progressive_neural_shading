# Phase 1.0: Minimal App framework
# 最简渲染框架: 窗口 + 设备 + blit
from typing import Optional
import slangpy as spy
from pathlib import Path


class App:
    """Minimal rendering application framework.

    Creates a window, a GPU device, and provides a blit() method
    to copy a Tensor to the screen.
    """

    def __init__(
        self,
        title: str = "Neural Shading",
        width: int = 1024,
        height: int = 1024,
    ):
        # Create window
        self._window = spy.Window(width=width, height=height, title=title, resizable=False)

        # Create GPU device
        self._device = spy.create_device(
            spy.DeviceType.automatic,
            enable_debug_layers=True,
            include_paths=[Path(__file__).parent]
        )

        # Load helper shaders (blit)
        self._module = spy.Module.load_from_file(self._device, "app.slang")

        # Setup swapchain
        self.surface = self._device.create_surface(self._window)
        self.surface.configure(width=width, height=height)

        # Output texture
        self._output_texture = self._device.create_texture(
            format=spy.Format.rgba16_float,
            width=width, height=height, mip_count=1,
            usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
            label="output_texture",
        )

        # ESC to close
        self._window.on_keyboard_event = self._on_keyboard_event

    @property
    def device(self) -> spy.Device:
        return self._device

    @property
    def window(self) -> spy.Window:
        return self._window

    def process_events(self) -> bool:
        """Returns False if window should close."""
        if self._window.should_close():
            return False
        self._window.process_events()
        return True

    def present(self):
        """Present the rendered frame to the screen."""
        if not self.surface.config:
            return
        image = self.surface.acquire_next_image()
        if not image:
            return

        command_encoder = self._device.create_command_encoder()
        command_encoder.blit(image, self._output_texture)
        command_encoder.set_texture_state(image, spy.ResourceState.present)
        self._device.submit_command_buffer(command_encoder.finish())
        del image
        self.surface.present()

    def blit(self, source: spy.Tensor, size: Optional[spy.int2] = None,
             offset: Optional[spy.int2] = None):
        """Copy a 2D Tensor to the output texture."""
        if size is None:
            size = spy.int2(source.shape[1], source.shape[0])
        if offset is None:
            offset = spy.int2(0, 0)
        self._module.blit(
            spy.grid((size.y, size.x)), size, offset, source, self._output_texture
        )

    def _on_keyboard_event(self, event: spy.KeyboardEvent):
        if event.type == spy.KeyboardEventType.key_press:
            if event.key == spy.KeyCode.escape:
                self._window.close()
