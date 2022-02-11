import matplotlib.pyplot as plt
from sklearn.manifold import TSNE 
import numpy as np 


class TSNE_VISUALIZE():
    def __init__(self):
        super().__init__()

    def plt2arr(self, fig):
        rgb_str = fig.canvas.tostring_rgb()
        (w,h) = fig.canvas.get_width_height()
        rgba_arr = np.fromstring(rgb_str, dtype=np.uint8, sep='').reshape((w,h,-1))
        return rgba_arr
    
    def visualize(self, out, color, epoch):
        fig = plt.figure(figsize=(5,5), frameon=False)
        fig.suptitle(f'Epoch = {epoch}')

        # Fit TSNE with 2 components
        z = TSNE(n_components=2).fit_transform(out.detach().cpu().numpy())

         # Create scatterplot from embeddings
        plt.xticks([])
        plt.yticks([])
        plt.scatter(z[:, 0], 
                    z[:, 1], 
                    s=70, 
                    c=color.detach().cpu().numpy(), 
                    cmap="Set2")
        fig.canvas.draw()

        # Convert to numpy
        rgba_arr = self.plt2arr(fig=fig)
        return rgba_arr