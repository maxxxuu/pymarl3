from .dmaq_qatten_learner import DMAQ_qattenLearner
from .nq_learner import NQLearner
from .nq_learner_data_augmentation import NQLearnerDataAugmentation
from .actor_critic_learner import ActorCriticLearner

REGISTRY = {}

REGISTRY["nq_learner"] = NQLearner
REGISTRY["dmaq_qatten_learner"] = DMAQ_qattenLearner
REGISTRY["q_learner_data_augmentation"] = NQLearnerDataAugmentation
REGISTRY["actor_critic_learner"] = ActorCriticLearner
