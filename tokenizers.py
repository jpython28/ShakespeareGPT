class BytePairEncoder:
  def __init__(self, max_tokens: int, corpus: str):
      self.max_tokens = max_tokens
      corpus_bytes = bytearray(corpus, encoding="ascii")
      self.token_pairs = {}
      while 128 + len(self.token_pairs) < self.max_tokens:
          print(f"\r{round((128 + len(self.token_pairs))/self.max_tokens*100.0, 3)}%", end="")
          frequencies = {}
          for i in range(len(corpus_bytes) - 1):
              pair = (corpus_bytes[i], corpus_bytes[i + 1])
              if pair in frequencies.keys():
                  frequencies[pair] += 1
              else:
                  frequencies[pair] = 1
          frequencies = dict(sorted(frequencies.items(), key=lambda x: x[1]))
          new_pair = list(frequencies.keys())[-1]
          self.token_pairs[new_pair] = 128 + len(self.token_pairs)
          new_corpus_bytes = []
          skip_next = False
          for j in range(len(corpus_bytes) - 1):
              pair = (corpus_bytes[j], corpus_bytes[j + 1])
              if not skip_next:
                  if pair == new_pair:
                      new_corpus_bytes.append(self.token_pairs[pair])
                      skip_next = True
                  else:
                      new_corpus_bytes.append(corpus_bytes[j])
              else:
                  skip_next = False
          if not skip_next:
              new_corpus_bytes.append(corpus_bytes[-1])
          corpus_bytes = new_corpus_bytes

  @property
  def reversed_token_pairs(self):
      return {v: k for k, v in self.token_pairs.items()}

  def encode(self, corpus):
      corpus_bytes = list(bytearray(corpus, encoding="ascii"))
      changed = True
      while changed:
          new_corpus_bytes = []
          skip_next = False
          changed = False
          for j in range(len(corpus_bytes) - 1):
              pair = (corpus_bytes[j], corpus_bytes[j + 1])
              if not skip_next:
                  if pair in self.token_pairs.keys():
                      new_corpus_bytes.append(self.token_pairs[pair])
                      skip_next = True
                      changed = True
                  else:
                      new_corpus_bytes.append(corpus_bytes[j])
              else:
                  skip_next = False
          if not skip_next:
              new_corpus_bytes.append(corpus_bytes[-1])
          corpus_bytes = new_corpus_bytes
      return corpus_bytes

  def decode(self, tokens: list[int]):
      corpus_bytes = tokens[:]
      while not all(map(lambda x: x < 128, corpus_bytes)):
          new_corpus_bytes = []
          for byte in corpus_bytes:
              if byte in self.reversed_token_pairs.keys():
                  pair = self.reversed_token_pairs[byte]
                  new_corpus_bytes.append(pair[0])
                  new_corpus_bytes.append(pair[1])
              else:
                  new_corpus_bytes.append(byte)
          corpus_bytes = new_corpus_bytes
      return "".join([i.to_bytes().decode("ascii") for i in corpus_bytes])