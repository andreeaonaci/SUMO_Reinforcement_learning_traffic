import torch
import torch.nn.functional as F

class NACRLAgent:
    def __init__(self, actor, critic, actor_optimizer, critic_optimizer):
        self.actor = actor
        self.critic = critic
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer

    def act(self, s_i):
        """
        Sample a discrete action from the policy.

        Args:
            s_i (torch.Tensor): State tensor, shape [state_dim]

        Returns:
            int: Discrete action (green phase index)
        """
        s_i = s_i.to(next(self.actor.parameters()).device)

        with torch.no_grad():
            dist = self.actor(s_i.unsqueeze(0))  # [1, action_dim]
            action = dist.sample()               # [1]

        return int(action.item())

    def update(self, batch, embedding, attention, gamma):
        for agent_id, transitions in batch.items():
            # unpack transitions
            s_i, a_i, r_i, s_i_next = zip(*transitions)

            s_i = torch.stack(s_i)                    # [B, obs_dim]
            a_i = torch.tensor(a_i).long()            # [B]
            r_i = torch.stack(r_i).squeeze(-1)        # [B]
            s_i_next = torch.stack(s_i_next)          # [B, obs_dim]

            # embeddings
            h_i = embedding(s_i)                      # [B, emb_dim]
            h_i_next = embedding(s_i_next)            # [B, emb_dim]

            # ---------- ATTENTION-BASED REWARD COORDINATION ----------
            if len(batch) > 1:
                h_others = []
                r_others = []

                for other_id, other_transitions in batch.items():
                    if other_id == agent_id:
                        continue

                    _, _, r_o, s_o_next = zip(*other_transitions)
                    r_others.append(torch.stack(r_o).squeeze(-1))        # [B]
                    h_others.append(embedding(torch.stack(s_o_next)))    # [B, emb_dim]

                h_others = torch.stack(h_others, dim=1)   # [B, N-1, emb_dim]
                r_others = torch.stack(r_others, dim=1)   # [B, N-1]

                alpha, _ = attention(h_i, h_others)       # alpha: [B, N-1]
                r_i_new = r_i + (alpha * r_others).sum(dim=1)
            else:
                r_i_new = r_i
            # --------------------------------------------------------

            # ---------- CRITIC UPDATE ----------
            V_next = self.critic(h_i_next).squeeze(-1)    # [B]
            y_i = r_i_new + gamma * V_next

            V = self.critic(h_i).squeeze(-1)              # [B]
            critic_loss = F.mse_loss(V, y_i.detach())

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            # -----------------------------------

            # ---------- ACTOR UPDATE ----------
            advantage = (y_i - V).detach()
            dist = self.actor(s_i)
            log_probs = dist.log_prob(a_i)
            actor_loss = -(log_probs * advantage).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            # ----------------------------------

