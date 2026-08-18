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

    @staticmethod
    def _embed_others(embedding, batch, agent_id, use_next_state):
        """Embeddings/rewards of every OTHER agent in `batch`, at the SAME
        timestep as the query embedding they'll be compared against
        (use_next_state=False for the current-state query h_i, True for the
        next-state query h_i_next) -- comparing h_i against other agents'
        NEXT-state embeddings was the original bug here: attention scores
        were computing similarity across two different timesteps.

        Returns (h_others [B, N-1, emb_dim], r_others [B, N-1]), or
        (None, None) if there are no other agents in this batch.
        """
        h_others, r_others = [], []
        for other_id, other_transitions in batch.items():
            if other_id == agent_id:
                continue
            s_o, _, r_o, s_o_next = zip(*other_transitions)
            s_o_at_t = s_o_next if use_next_state else s_o
            h_others.append(embedding(torch.stack(s_o_at_t)))
            r_others.append(torch.stack(r_o).squeeze(-1))

        if not h_others:
            return None, None
        return torch.stack(h_others, dim=1), torch.stack(r_others, dim=1)

    def update(self, agent_id, batch, embedding, attention, embedding_optimizer, gamma):
        """Update THIS agent's actor/critic using only its OWN transitions
        (batch[agent_id]).

        Other agents' entries in `batch` are used only as context for the
        attention-based reward/embedding coordination below -- never to
        directly train this agent's actor/critic on another agent's
        (state, action) pairs. The original version of this method looped
        over every agent_id in `batch` and ran a full actor/critic update
        step for each one using `self.actor`/`self.critic` (this specific
        agent's networks) on every OTHER agent's transitions too -- meaning
        every agent's networks got fit to reproduce every other agent's
        behavior each round, not just its own.
        """
        if agent_id not in batch:
            return

        s_i, a_i, r_i, s_i_next = zip(*batch[agent_id])
        s_i = torch.stack(s_i)                    # [B, obs_dim]
        a_i = torch.tensor(a_i).long()             # [B]
        r_i = torch.stack(r_i).squeeze(-1)         # [B]
        s_i_next = torch.stack(s_i_next)           # [B, obs_dim]

        # embeddings
        h_i = embedding(s_i)                       # [B, emb_dim]
        h_i_next = embedding(s_i_next)              # [B, emb_dim]

        # ---------- ATTENTION-BASED REWARD & EMBEDDING COORDINATION ------
        h_others, r_others = self._embed_others(embedding, batch, agent_id, use_next_state=False)
        h_others_next, _ = self._embed_others(embedding, batch, agent_id, use_next_state=True)

        if h_others is not None:
            alpha, h_i_new = attention(h_i, h_others)
            r_i_new = r_i + (alpha * r_others).sum(dim=1)
            _, h_i_next_new = attention(h_i_next, h_others_next)
        else:
            # Single-agent batch: nothing to attend over, fall back to the
            # plain embeddings/reward untouched.
            r_i_new = r_i
            h_i_new = h_i
            h_i_next_new = h_i_next
        # ------------------------------------------------------------------

        # ---------- CRITIC UPDATE ----------
        # Uses the context-aware embeddings (h_i_new/h_i_next_new) that
        # AgentAttention actually produces -- previously discarded (only the
        # attention weights `alpha` were kept), so the critic never saw the
        # coordinated representation the attention module was built to
        # provide, and the embedding network never received a gradient at
        # all (no optimizer existed for it, so its parameters were random
        # and frozen for the agent's entire lifetime). embedding_optimizer
        # is stepped here for that reason -- it's the only path any
        # gradient reaches the shared embedding network through.
        V_next = self.critic(h_i_next_new).squeeze(-1)    # [B]
        y_i = r_i_new + gamma * V_next

        V = self.critic(h_i_new).squeeze(-1)               # [B]
        critic_loss = F.mse_loss(V, y_i.detach())

        self.critic_optimizer.zero_grad()
        embedding_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        embedding_optimizer.step()
        # -----------------------------------

        # ---------- ACTOR UPDATE ----------
        # Actor operates on raw state s_i (Actor.forward), not the shared
        # embedding, so it needs no embedding_optimizer step.
        advantage = (y_i - V).detach()
        dist = self.actor(s_i)
        log_probs = dist.log_prob(a_i)
        actor_loss = -(log_probs * advantage).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        # ----------------------------------
