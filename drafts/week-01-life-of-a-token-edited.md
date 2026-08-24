# The Life of a Token

### Inference from Scratch #1 - What actually happens when an LLM generates a token?

*This is the first post in **Inference from Scratch** series, my attempt to learn LLM inference in public, from a naive PyTorch loop all the way to a fully optimized inference engine. I'm not an expert in LLM inference, so please feel free to correct me if I get something wrong! The deal is that nothing appears here unless I've derived it, run it, or measured it, and when I get something wrong I'll correct it in the open. The goal is to publish a new deep dive every week and a separate weekly digest, This Week in Inference, covers what changes in the field each week.*

---

During the early days of ChatGPT, if you ever types a prompt and hit enter, you might have noticed that the first token of the response would take a long time to appear. Then, after that, the rest of the tokens would appear much faster, even though the model "read" you entire prompt almost instantly. Why is reading 1,000 words so fast, but writing the 1,001st slow?

Since the paper on transformers was published in 2017, I have worked with many different transformer models, and even trained and finetuned some of the "larger" models (upto 70b models) for real world applications at work. But most of those uses cases were for either classification or embedding generation tasks. I never really thought of the inference process in terms of token generation and the challenges that come with it. Until recently, I was under the impression that the expensive part of LLM is training and that inference is just a simple forward pass through the model. But as I tried to use open LLMs to build a personal assistant, I realized that inference is not as simple as I thought. And as I dove deep into the inference rabit hole, I was completely overwhelemed by the amount of research out there, but not enough of a structured resource for someone like me to learn from scratch. This series is the one I wish I'd read when I started learning about LLM inference. We'll build the dumbest possible inference loop in ~50 lines of PyTorch, run it, and derive a one-line formula that predicts LLM generation speed  on hardware from a MacBook to an H100 GPU. Weather the prediction survives contact with real silicon if later weeks job, and I'll publish the misses along with the hits.

