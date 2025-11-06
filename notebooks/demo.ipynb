import torch
from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Callable

class AdaptiveSGQ:
    def __init__(self,
                 functions: List[Callable],
                 learning_rate: float = 0.01,
                 initial_p: float = 0.2,
                 min_p: float = 0.1,
                 max_p: float = 0.5,
                 switch_threshold: float = 0.001,
                 L: float = 1.0):

        self.functions = functions
        self.n_functions = len(functions)
        self.lr = learning_rate
        self.current_p = initial_p
        self.min_p = min_p
        self.max_p = max_p
        self.switch_threshold = switch_threshold
        self.L = L

        self.x = torch.randn(5, requires_grad=True)
        self.iteration = 0
        self.best_loss = float('inf')

        self.loss_history = []
        self.p_history = []
        self.mode_history = []
        self.algorithm_mode = 'SGQ'

        print(f'Initialized adaptive SGQ with p={initial_p} & mode={self.algorithm_mode}')

    def compute_ei(self, gradient: Tensor, full_gradient: Tensor) -> float:
        alpha = self.lr
        ei = (alpha * torch.dot(full_gradient, gradient) -
              (alpha ** 2 * self.L / 2) * torch.norm(gradient)**2)
        return ei.item()

    def Adaptive_p_update(self, current_loss: float, improvement: float):
      if improvement > 0.05:
        self.current_p = max(self.min_p, self.current_p * 0.9)

      elif improvement < 0.005:
        self.current_p = min(self.max_p, self.current_p * 1.1)

      if (current_loss < self.switch_threshold and self.algorithm_mode and
        self.algorithm_mode == 'SGQ' and self.iteration > 100):

        self.algorithm_mode = 'SGD'
        print(f'switching to SGD at iteration {self.iteration} (loss: {current_loss:.6f})')

    def strategic_selection(self) -> int:
      gradients= []
      for i in range(self.n_functions):
        self.x.grad = None
        loss = self.functions[i](self.x)
        loss.backward()
        if self.x.grad is not None:
          gradients.append(self.x.grad.clone())
        else:
          gradients.append(torch.zeros_like(self.x))

      full_gradient = torch.mean(torch.stack(gradients), dim= 0)

      if np.random.random() < self.current_p:
        selected_index = np.random.randint(self.n_functions)
        return selected_index

      eis = []
      for grad in gradients:
        ei = self.compute_ei(grad, full_gradient)
        eis.append(ei)

      selected_index = np.argmax(eis)

      return selected_index

    def compute_current_loss(self) -> float:
      total_loss = 0
      for function in self.functions:
        total_loss += function(self.x).item()
      return total_loss / self.n_functions

    def optimize_step(self):
      if self.algorithm_mode == 'SGQ':
        #sgq phase
        selected_index = self.strategic_selection()
      else:
        #sgd phase
        selected_index = np.random.randint(self.n_functions)

      self.x.grad = None
      loss = self.functions[selected_index](self.x)
      loss.backward()

      with torch.no_grad():
        if self.x.grad is not None:
          self.x -= self.lr * self.x.grad

      current_loss = self.compute_current_loss()
      improvement = self.best_loss - current_loss if self.best_loss != float('inf') else 0

      if current_loss < self.best_loss:
        self.best_loss = current_loss

      #adaptive p update(sgq)
      if self.algorithm_mode == 'SGQ':
        self.Adaptive_p_update(current_loss, improvement)

      #history
      self.loss_history.append(current_loss)
      self.p_history.append(self.current_p)
      self.mode_history.append(self.algorithm_mode)
      self.iteration += 1

      return current_loss

    def optimize(self, iterations: int = 1000, verbose: bool = True):
      print(f'starting optimization with {self.algorithm_mode}')
      print(f'initial p: {self.current_p:.3f}')
      print(f'switch threshold: {self.switch_threshold}')

      for i in range(iterations):
        loss = self.optimize_step()

        if verbose and i % 100 == 0:
          mode_indirector = '🔵' if self.algorithm_mode == 'SGQ' else '🔴'
          print(f'{mode_indirector} iteration {i}: loss ={loss:.6f}, p = {self.current_p:.3f}')

        # early stopping
        if loss < 1e-6:
          print(f'coverged at iteration {i}')
          break

        final_mode = 'SGD' if self.algorithm_mode == 'SGD' else 'SGQ'
        print(f'optimization completed in {self.iteration} iterations')
        print(f'final mode: {final_mode}')
        print(f'final loss: {self.loss_history[-1]:.6f}')
        print(f'final p: {self.p_history[-1]:.3f}')

def plot_results(self, save_path: str = 'adaptive_sgq_results.png'):
      fig, ((ax1, ax2), (ax3, ax4))= plt.subplot(2, 2, figsize=(12,8))

      #loss history
      ax1.semilogy(self.loss_history)
      ax1.set_xlabel('iteration')
      ax1.set_ylabel('loss')
      ax1.set_title('optimization progress')
      ax1.grid(True, alpha=0.3)

      #algorithm switch
      if 'SGD' in self.mode_history:
        switch_iter = self.mode_history.index('SGD')
        ax1.axvline(x=switch_iter, color='red', linestyle='--', alpha=0.7, label='sgq->sgd')
        ax1.legend()

      #p adaption
      ax2.semilogy(self.p_history)
      ax2.set_xlabel('iteration')
      ax2.set_ylabel('exploration probability (p)')
      ax2.set_title('adaptive p parameter')
      ax2.grid(True, alpha=0.3)

      #mode history
      mode_numeric = [1 if mode == 'SGQ' else 0 for mode in self.mode_history]
      ax3.plot(mode_numeric)
      ax3.set_xlabel('iteration')
      ax3.set_ylabel('algorithm mode')
      ax3.set_title('sgq (1) vs sgd(0) mode')
      ax3.set_yticks([0, 1])
      ax3.set_yticklabels(['SGD', 'SGQ'])
      ax3.grid(True, alpha=0.3)

      #improvement rate
      improvements = [0] + [-self.loss_history[i] + self.loss_history[i-1] for i in range(len(self.loss_history))]
      ax4.plot(improvements)
      ax4.set_xlabel('iteration')
      ax4.set_ylabel('improvement rate')
      ax4.set_title('loss improvement per iteration')
      ax4.grid(True, alpha=0.3)

      plt.tight_layout()
      plt.savefig(save_path, dpi=300, bbox_inches='tight')
      plt.show()

      return fig

# example
def create_heterogeneous_functions(n_functions: int = 8):
  functions=[]

  centers = [
      torch.tensor([1.0, 1.0, 0.5, -0.5, 0.0]),
      torch.tensor([-1.0, 2.0, -0.5, 1.0, 0.5]),
      torch.tensor([0.5, -1.0, 1.0, -1.0, 1.0]),
      torch.tensor([-2.0, -0.5, 0.0, 0.5, -1.0]),
      torch.tensor([1.5, 0.0, -1.0, 1.5, 0.5]),
      torch.tensor([-1.0, -2.0, 0.5, -0.5, 1.0]),
      torch.tensor([0.0, 1.5, -1.5, 1.0, -0.5]),
      torch.tensor([2.0, -1.0, 1.0, -1.5, 0.0]),
  ]

  for i, center in enumerate(centers[:n_functions]):
    scale = 0.5 + 0.3 * (i % 3)

    def make_func(c=center.clone(), s=scale):
      def func(x):
        return s * torch.norm(x - c)**2
      return func

    functions.append(make_func())

  return functions

  #demo execution
  if __name__ == '__main__':
    test_functions = create_heterogeneous_functions()

    optimizer = AdaptiveSGQ(
        functions = test_functions,
        learning_rate = 0.1,
        initial_p = 0.2,
        switch_threshold = 0.01
        )

    optimizer.optimize(iterations=500)

    optimizer.plot_results()
