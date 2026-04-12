from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pygame


@dataclass
class RenderConfig:
    width: int = 480
    height: int = 640
    fps: int = 60
    sprite_path: str | None = None


class PygameRenderer:
    def __init__(self, config: RenderConfig | None = None):
        self.config = config or RenderConfig()
        pygame.init()
        self.screen = pygame.display.set_mode((self.config.width, self.config.height))
        pygame.display.set_caption("Flappy Bird RL")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Trebuchet MS", 28, bold=True)
        self.small_font = pygame.font.SysFont("Trebuchet MS", 18)
        self.frame_idx = 0
        self.bird_sprite = self._load_bird_sprite()

    def _candidate_sprite_paths(self) -> list[Path]:
        custom = Path(self.config.sprite_path) if self.config.sprite_path else None
        options: list[Path] = []
        if custom is not None:
            options.append(custom)
        options.append(Path.cwd() / "flappy.png")
        options.append(Path(__file__).resolve().parents[1] / "flappy.png")
        return options

    def _load_bird_sprite(self) -> pygame.Surface | None:
        for p in self._candidate_sprite_paths():
            if p.exists():
                sprite = pygame.image.load(str(p)).convert()
                sprite.set_colorkey((0, 0, 0))
                return sprite
        return None

    def _draw_sky(self, w: int, h: int):
        horizon = int(h * 0.86)
        for y in range(horizon):
            t = y / max(1, horizon)
            r = int(130 + 90 * t)
            g = int(205 + 30 * t)
            b = int(250 - 20 * t)
            pygame.draw.line(self.screen, (r, g, b), (0, y), (w, y))

        sun_x = int(w * 0.78)
        sun_y = int(h * 0.17)
        pygame.draw.circle(self.screen, (255, 225, 140), (sun_x, sun_y), 46)
        pygame.draw.circle(self.screen, (255, 243, 190), (sun_x, sun_y), 32)

        # Soft animated clouds.
        drift = self.frame_idx * 0.5
        cloud_specs = [
            (int((60 + drift) % (w + 120)) - 120, 95, 42),
            (int((240 + drift * 0.8) % (w + 150)) - 150, 155, 34),
            (int((420 + drift * 0.6) % (w + 180)) - 180, 70, 28),
        ]
        for cx, cy, r in cloud_specs:
            pygame.draw.circle(self.screen, (245, 250, 255), (cx, cy), r)
            pygame.draw.circle(self.screen, (245, 250, 255), (cx + r // 2, cy + 8), int(r * 0.8))
            pygame.draw.circle(self.screen, (245, 250, 255), (cx - r // 2, cy + 8), int(r * 0.8))

    def _draw_ground(self, w: int, h: int):
        ground_y = int(h * 0.86)
        pygame.draw.rect(self.screen, (193, 144, 84), (0, ground_y, w, h - ground_y))
        stripe_w = 40
        offset = (self.frame_idx * 4) % stripe_w
        for x in range(-stripe_w, w + stripe_w, stripe_w):
            pygame.draw.rect(
                self.screen,
                (210, 162, 95),
                (x - offset, ground_y + 5, stripe_w // 2, h - ground_y - 5),
            )
        pygame.draw.line(self.screen, (119, 83, 44), (0, ground_y), (w, ground_y), 4)

    def draw(self, state: dict):
        w = self.config.width
        h = self.config.height
        self.frame_idx += 1

        self._draw_sky(w, h)

        bird_x = int(float(state["bird_x"]) * w)
        bird_y = int((1.0 - float(state["bird_y"])) * h)
        bird_radius = max(5, int(float(state["bird_radius"]) * w))

        pipe_x = int(float(state["pipe_x"]) * w)
        pipe_w = max(8, int(float(state["pipe_width"]) * w))

        gap_center = float(state["gap_center"])
        gap_half = float(state["gap_half_height"])
        gap_top_y = int((1.0 - (gap_center + gap_half)) * h)
        gap_bottom_y = int((1.0 - (gap_center - gap_half)) * h)

        ground_y = int(h * 0.86)
        top_rect = pygame.Rect(pipe_x, 0, pipe_w, max(0, gap_top_y))
        bottom_rect = pygame.Rect(pipe_x, min(ground_y, gap_bottom_y), pipe_w, max(0, ground_y - gap_bottom_y))

        pipe_color = (53, 176, 92)
        pipe_shadow = (34, 120, 62)
        for rect in (top_rect, bottom_rect):
            pygame.draw.rect(self.screen, pipe_shadow, rect)
            inner = rect.inflate(-6, 0)
            pygame.draw.rect(self.screen, pipe_color, inner)
            cap_h = 14
            cap = pygame.Rect(rect.x - 6, rect.bottom - cap_h, rect.width + 12, cap_h)
            if rect is top_rect:
                cap = pygame.Rect(rect.x - 6, rect.bottom - cap_h, rect.width + 12, cap_h)
            else:
                cap = pygame.Rect(rect.x - 6, rect.y, rect.width + 12, cap_h)
            pygame.draw.rect(self.screen, (75, 195, 110), cap)
            pygame.draw.rect(self.screen, (32, 110, 58), cap, 2)

        if self.bird_sprite is not None:
            sprite_size = int(bird_radius * 3.2)
            sprite = pygame.transform.smoothscale(self.bird_sprite, (sprite_size, sprite_size))
            rect = sprite.get_rect(center=(bird_x, bird_y))
            self.screen.blit(sprite, rect)
        else:
            pygame.draw.circle(self.screen, (255, 215, 0), (bird_x, bird_y), bird_radius)

        self._draw_ground(w, h)

        panel = pygame.Surface((160, 62), pygame.SRCALPHA)
        panel.fill((255, 255, 255, 175))
        self.screen.blit(panel, (10, 10))
        score_text = self.font.render(f"Score: {int(state['score'])}", True, (18, 44, 36))
        self.screen.blit(score_text, (20, 20))
        hint_text = self.small_font.render("Press Q to quit", True, (40, 64, 58))
        self.screen.blit(hint_text, (20, 48))

        pygame.display.flip()
        self.clock.tick(self.config.fps)

    def handle_quit(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYDOWN and event.key in (pygame.K_q, pygame.K_ESCAPE):
                return True
        return False

    def close(self):
        pygame.quit()
