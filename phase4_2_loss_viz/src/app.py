# Phase 1.2: App framework with tonemap support
from typing import Optional
import slangpy as spy
from pathlib import Path


class App:
    def __init__(self, title="Neural Shading", width=1024, height=1024):
        self._window = spy.Window(width=width, height=height, title=title, resizable=False)
        self._device = spy.create_device(
            spy.DeviceType.automatic, enable_debug_layers=True,
            include_paths=[Path(__file__).parent]
        )
        self._module = spy.Module.load_from_file(self._device, "app.slang")
        self.surface = self._device.create_surface(self._window)
        self.surface.configure(width=width, height=height)
        self._output_texture = self._device.create_texture(
            format=spy.Format.rgba16_float, width=width, height=height, mip_count=1,
            usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
            label="output_texture",
        )
        self._window.on_keyboard_event = self._on_keyboard_event

    @property
    def device(self): return self._device
    @property
    def window(self): return self._window

    def process_events(self):
        if self._window.should_close(): return False
        self._window.process_events(); return True

    def present(self):
        if not self.surface.config: return
        image = self.surface.acquire_next_image()
        if not image: return
        cmd = self._device.create_command_encoder()
        cmd.blit(image, self._output_texture)
        cmd.set_texture_state(image, spy.ResourceState.present)
        self._device.submit_command_buffer(cmd.finish())
        del image; self.surface.present()

    def blit(self, source, size=None, offset=None, tonemap=True):
        if size is None: size = spy.int2(source.shape[1], source.shape[0])
        if offset is None: offset = spy.int2(0, 0)
        self._module.blit(
            spy.grid((size.y, size.x)), size, offset, tonemap, source, self._output_texture
        )

    def _on_keyboard_event(self, event):
        if event.type == spy.KeyboardEventType.key_press:
            if event.key == spy.KeyCode.escape: self._window.close()
