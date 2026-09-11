from pathlib import Path

from jinja2 import BaseLoader, Environment, FileSystemLoader

from .loader import Sample


class PromptTemplater:
    def __init__(self, template_dir: str | Path | None = None, template_str: str | None = None):
        if template_str is not None:
            self._env = Environment(loader=BaseLoader())
            self._default_tmpl = self._env.from_string(template_str)
        elif template_dir is not None:
            self._env = Environment(loader=FileSystemLoader(str(template_dir)))
            self._default_tmpl = None
        else:
            raise ValueError("Provide template_dir or template_str")

    def apply(self, sample: Sample, template_name: str | None = None) -> str:
        tmpl = self._default_tmpl or self._env.get_template(template_name)
        return tmpl.render(sample=sample, **sample.metadata).strip()

    def apply_batch(self, samples: list[Sample], template_name: str | None = None) -> list[str]:
        return [self.apply(s, template_name) for s in samples]
