import numpy as np
import matplotlib.pyplot as plt

class GrenanderDensity:
    def __init__(self, x):
        x = np.sort(np.asarray(x))
        n = len(x)

        dx = np.diff(x)
        dx[dx <= 0] = 1e-12  # avoid division by zero

        slopes = (1.0 / n) / dx


        heights = slopes.tolist()
        widths = dx.tolist()
        lefts = x[:-1].tolist()
        rights = x[1:].tolist()

        i = 0
        while i < len(heights) - 1:
            if heights[i] < heights[i+1]:  # violation
                total_width = widths[i] + widths[i+1]
                pooled_height = (
                    heights[i]*widths[i] +
                    heights[i+1]*widths[i+1]
                ) / total_width

                heights[i] = pooled_height
                widths[i] = total_width
                rights[i] = rights[i+1]

                del heights[i+1]
                del widths[i+1]
                del lefts[i+1]
                del rights[i+1]

                if i > 0:
                    i -= 1
            else:
                i += 1

        self.lefts = np.array(lefts)
        self.rights = np.array(rights)
        self.heights = np.array(heights)


    def __call__(self, t):
        """
        Evaluate density at points t
        """
        t = np.asarray(t)
        result = np.zeros_like(t, dtype=float)


        for i in range(len(self.heights)):
            mask = (t >= self.lefts[i]) & (t < self.rights[i])
            result[mask] = self.heights[i]

        return result

