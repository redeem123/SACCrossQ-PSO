with open("sage_latex_template_4_unzipped/references.bib", "r") as f:
    content = f.read()

content = content.replace("""@misc{Zhang2024DDPGPSO,
  note = {Zhang, Y., Liu, H., Wang, X.
(2024).
``\\href{https://doi.org/10.1016/j.swevo.2024.101458}{Particle swarm optimization algorithm based on deep deterministic policy gradient,}''
\\textit{Swarm and Evolutionary Computation}, 85, 101458.}
}""", """@article{Zhang2024DDPGPSO,
  author = {Zhang, Y. and Liu, H. and Wang, X.},
  title = {Particle swarm optimization algorithm based on deep deterministic policy gradient},
  journal = {Swarm and Evolutionary Computation},
  volume = {85},
  pages = {101458},
  year = {2024},
  doi = {10.1016/j.swevo.2024.101458}
}""")

content = content.replace("""@misc{Bhatt2024CrossQ,
  note = {Bhatt, A., Palenicek, D., Belousov, B., Argus, M., Amiranashvili, A., Brox, T., Peters, J.
(2024).
``\\href{https://doi.org/10.48550/arXiv.1902.05605}{CrossQ: Batch normalization in deep reinforcement learning for greater sample efficiency and simplicity,}''
\\textit{arXiv:1902.05605}.}
}""", """@article{Bhatt2024CrossQ,
  author = {Bhatt, A. and Palenicek, D. and Belousov, B. and Argus, M. and Amiranashvili, A. and Brox, T. and Peters, J.},
  title = {CrossQ: Batch normalization in deep reinforcement learning for greater sample efficiency and simplicity},
  journal = {arXiv preprint arXiv:1902.05605},
  year = {2024}
}""")

with open("sage_latex_template_4_unzipped/references.bib", "w") as f:
    f.write(content)
