# EDEN OS V2 — Credits & Acknowledgements

## Hallo — Portrait Image Animation

EDEN OS V2's face animation system is powered by **Hallo**, developed by the
**Fudan University Generative Vision Lab** (fudan-generative-vision).

We extend our deepest gratitude and thanks to the Hallo team for their
groundbreaking research in audio-driven portrait animation.

### Hallo4: High-Fidelity Dynamic Portrait Animation via Direct Preference Optimization
- **Paper**: SIGGRAPH Asia 2025
- **Authors**: Jiahao Cui, Baoyou Chen, Mingwang Xu, Hanlin Shang, Yuxuan Chen,
  Yun Zhan, Zilong Dong, Yao Yao, Jingdong Wang, Siyu Zhu
- **Institutions**: Fudan University, Baidu Inc, Nanjing University, Alibaba Group
- **Repository**: https://github.com/fudan-generative-vision/hallo4
- **Model**: https://huggingface.co/fudan-generative-ai/hallo4
- **arXiv**: https://arxiv.org/abs/2505.23525

### Hallo3: Highly Dynamic and Realistic Portrait Image Animation with Diffusion Transformer Networks
- **Authors**: Jiahao Cui, Hui Li, Yun Zhan, et al.
- **Repository**: https://github.com/fudan-generative-vision/hallo3
- **Model**: https://huggingface.co/fudan-generative-ai/hallo3
- **arXiv**: https://arxiv.org/abs/2412.00733

### Citation

```bibtex
@misc{cui2025hallo4,
    title={Hallo4: High-Fidelity Dynamic Portrait Animation via Direct Preference Optimization},
    author={Jiahao Cui and Baoyou Chen and Mingwang Xu and Hanlin Shang and
            Yuxuan Chen and Yun Zhan and Zilong Dong and Yao Yao and
            Jingdong Wang and Siyu Zhu},
    year={2025},
    eprint={2505.23525},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}

@misc{cui2024hallo3,
    title={Hallo3: Highly Dynamic and Realistic Portrait Image Animation
           with Diffusion Transformer Networks},
    author={Jiahao Cui and Hui Li and Yun Zhang and Hanlin Shang and
            Kaihui Cheng and Yuqi Ma and Shan Mu and Hang Zhou and
            Jingdong Wang and Siyu Zhu},
    year={2024},
    eprint={2412.00733},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

## Additional Technologies

- **Edge TTS** — Microsoft Edge Text-to-Speech for Eve's voice
- **AvatarForcing** — One-step streaming talking avatars (arXiv:2603.14331)
- **Wav2Vec2** — Facebook's audio encoder (facebook/wav2vec2-base-960h)
- **WAN2.1** — Base video generation model (Wan-AI/Wan2.1-T2V-1.3B)
- **MediaPipe** — Google's face mesh detection

## License

Hallo4 is a derivative of WAN2.1-1.3B, governed by the WAN LICENSE.
Hallo3 is a derivative of CogVideo-5B, released under MIT license.
