from torch import nn

class CLIP2GPT(nn.Module):
    def __init__(self, clip_dim, gpt_dim):
        super(CLIP2GPT, self).__init__()
        self.fc = nn.Linear(clip_dim, gpt_dim)

    def forward(self, clip_features):
        return self.fc(clip_features)
