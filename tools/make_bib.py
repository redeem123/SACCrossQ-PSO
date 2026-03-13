import re
import urllib.request
import urllib.error
import time

tex_file = "sage_latex_template_4_unzipped/main.tex"
with open(tex_file, "r") as f:
    content = f.read()

bib_block_match = re.search(r'\\begin{thebibliography}{.*?}(.*?)\\end{thebibliography}', content, re.DOTALL)
if not bib_block_match:
    print("Could not find thebibliography block")
    exit(1)

bib_block = bib_block_match.group(1)

items = re.split(r'\\bibitem{([^}]+)}', bib_block)
items = items[1:] # first is empty

bib_entries = []

manual_bibs = {
    "Sutton2018Reinforcement": """@book{Sutton2018Reinforcement,
  author = {Sutton, R. S. and Barto, A. G.},
  title = {Reinforcement Learning: An Introduction},
  publisher = {MIT Press},
  edition = {2nd},
  year = {2018},
  url = {http://incompleteideas.net/book/the-book-2nd.html}
}""",
    "Bai2023Evolutionary": """@article{Bai2023Evolutionary,
  author = {Bai, H. and Cheng, R. and Jin, Y.},
  title = {Evolutionary reinforcement learning: A survey},
  journal = {arXiv preprint arXiv:2108.11863},
  year = {2021}
}""",
    "Lillicrap2016Continuous": """@inproceedings{Lillicrap2016Continuous,
  author = {Lillicrap, T. P. and Hunt, J. J. and Pritzel, A. and Heess, N. and Erez, T. and Tassa, Y. and Silver, D. and Wierstra, D.},
  title = {Continuous control with deep reinforcement learning},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year = {2016}
}""",
    "Fujimoto2018Addressing": """@inproceedings{Fujimoto2018Addressing,
  author = {Fujimoto, S. and van Hoof, H. and Meger, D.},
  title = {Addressing function approximation error in actor-critic methods},
  booktitle = {Proceedings of the 35th International Conference on Machine Learning},
  pages = {1587--1596},
  year = {2018}
}""",
    "Lange2024Evolution": """@article{Lange2024Evolution,
  author = {Lange, R. T. and Schaul, T. and Chen, Y. and Zahavy, T. and Dalibard, V. and Lu, C. and Singh, S. and Flennerhag, S.},
  title = {Evolution Transformer: In-context evolutionary optimization},
  journal = {arXiv preprint arXiv:2403.02985},
  year = {2024}
}""",
    "Chen2021Decision": """@inproceedings{Chen2021Decision,
  author = {Chen, L. and Lu, K. and Rajeswaran, A. and Lee, K. and Grover, A. and Laskin, M. and Abbeel, P. and Srinivas, A. and Mordatch, I.},
  title = {Decision Transformer: Reinforcement Learning via Sequence Modeling},
  booktitle = {Advances in Neural Information Processing Systems},
  volume = {34},
  pages = {15084--15097},
  year = {2021}
}""",
    "Schulman2017Proximal": """@article{Schulman2017Proximal,
  author = {Schulman, J. and Wolski, F. and Dhariwal, P. and Radford, A. and Klimov, O.},
  title = {Proximal policy optimization algorithms},
  journal = {arXiv preprint arXiv:1707.06347},
  year = {2017}
}""",
    "Ng1999Policy": """@inproceedings{Ng1999Policy,
  author = {Ng, A. Y. and Harada, D. and Russell, S.},
  title = {Policy invariance under reward transformations: Theory and application to reward shaping},
  booktitle = {Proceedings of the 16th International Conference on Machine Learning},
  pages = {278--287},
  year = {1999}
}""",
    "Sener2018MultiTask": """@inproceedings{Sener2018MultiTask,
  author = {Sener, O. and Koltun, V.},
  title = {Multi-Task Learning as Multi-Objective Optimization},
  booktitle = {Advances in Neural Information Processing Systems},
  volume = {31},
  pages = {527--538},
  year = {2018}
}""",
    "Haarnoja2018Soft": """@inproceedings{Haarnoja2018Soft,
  author = {Haarnoja, T. and Zhou, A. and Abbeel, P. and Levine, S.},
  title = {Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor},
  booktitle = {Proceedings of the 35th International Conference on Machine Learning},
  pages = {1861--1870},
  year = {2018}
}""",
    "Ioffe2015Batch": """@inproceedings{Ioffe2015Batch,
  author = {Ioffe, S. and Szegedy, C.},
  title = {Batch normalization: Accelerating deep network training by reducing internal covariate shift},
  booktitle = {Proceedings of the 32nd International Conference on Machine Learning},
  pages = {448--456},
  year = {2015}
}""",
    "Vaswani2017Attention": """@inproceedings{Vaswani2017Attention,
  author = {Vaswani, A. and Shazeer, N. and Parmar, N. and Uszkoreit, J. and Jones, L. and Gomez, A. N. and Kaiser, L. and Polosukhin, I.},
  title = {Attention is all you need},
  booktitle = {Advances in Neural Information Processing Systems},
  volume = {30},
  pages = {5998--6008},
  year = {2017}
}"""
}

for i in range(0, len(items), 2):
    cite_key = items[i]
    text = items[i+1]
    
    # Try to find DOI
    doi_match = re.search(r'(?:doi\.org/|doi=|doi:|doi.org\\/)(10\.\d{4,9}/[-._;()/:A-Z0-9]+)', text, re.IGNORECASE)
    if not doi_match:
        doi_match = re.search(r'arXiv\.(\d{4}\.\d{5})', text, re.IGNORECASE)
    
    if cite_key in manual_bibs:
        bib_entries.append(manual_bibs[cite_key])
        print(f"[{cite_key}] manually added")
        continue

    bibtex_str = None
    if doi_match:
        doi = doi_match.group(1).rstrip('.')
        if 'arXiv' not in text:
            # fetch bibtex from crossref
            url = f"https://api.crossref.org/works/{doi}/transform/application/x-bibtex"
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req) as response:
                    bibtex_str = response.read().decode('utf-8')
                    # Replace the key returned by crossref with our cite_key
                    bibtex_str = re.sub(r'@[a-zA-Z]+\{.*?,', lambda m: m.group(0).split('{')[0] + '{' + cite_key + ',', bibtex_str, count=1)
                    print(f"[{cite_key}] crossref success")
            except Exception as e:
                print(f"[{cite_key}] crossref failed for {doi}: {e}")
                
    if not bibtex_str:
        # fallback manual format
        print(f"[{cite_key}] fallback formatting needed")
        bibtex_str = f"@misc{{{cite_key},\n  note = {{{text.strip()}}}\n}}"
    
    bib_entries.append(bibtex_str)
    time.sleep(0.1) # be nice to crossref API

with open("sage_latex_template_4_unzipped/references.bib", "w") as f:
    f.write("\n\n".join(bib_entries))
print("references.bib written!")
