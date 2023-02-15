import time
import random
from typing import Optional, Union
import numpy as np
from negmas.sao.negotiators import SAONegotiator, NaiveTitForTatNegotiator
from sklearn.gaussian_process import GaussianProcessRegressor
import matplotlib.pyplot as plt

from negmas import Controller, MechanismState, ResponseType, SAOMechanism


class RandomNegotiator(SAONegotiator):
    """Here is an example of a random negotiator agent class"""

    def __init__(self, name: str = None, parent: Controller = None, ufun: Optional["UtilityFunction"] = None,
                 initial_concession: Union[float, str] = "min", **kwargs):
        super().__init__(name=name, ufun=ufun, parent=parent, **kwargs)
        self.initial_concession = initial_concession  # necessary

    def on_ufun_changed(self):
        """Method to instantiate ordered outcomes attribute, useful for utility functions."""
        super().on_ufun_changed()
        outcomes = self._ami.discrete_outcomes()
        self.ordered_outcomes = sorted(
            [(self._utility_function(outcome), outcome) for outcome in outcomes],
            key=lambda x: x[0],
            reverse=True,
        )

    def respond(self, state: MechanismState, offer: "Outcome") -> "ResponseType":
        """Method called when the agent receives the offer "offer";
        it must return either ACCEPT_OFFER or REJECT_OFFER. You also can access the offers"""
        print(offer)
        if random.random() < 0.5:
            return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER

    def propose(self, state: MechanismState) -> Optional["Outcome"]:
        """Method called when the agent needs to make an offer on
        1st round or later. Should send a proposal."""
        offer = random.choice(self._ami.discrete_outcomes())
        return offer


class CustomNegotiator(SAONegotiator):  # Here is an example of a negotiator agent class
    def __init__(self,n_steps, name: str = None, parent: Controller = None, ufun: Optional["UtilityFunction"] = None,
                 initial_concession: Union[float, str] = "min", **kwargs):
        super().__init__(name=name, ufun=ufun, parent=parent, **kwargs)
        self.initial_concession = initial_concession
        self.past_offers = []
        self.t = 0
        self.N_utility = 0.
        self.T=n_steps
        self.upper_u_threshold = None
        if self.ufun!=None:
            self.upper_u_threshold = np.max(self.ufun)*0.9
            self.lower_u_threshold = np.max(self.ufun)*0.3

    def on_ufun_changed(self):
        """Method to instantiate ordered outcomes attribute, useful for utility functions."""
        super().on_ufun_changed()
        outcomes = self._ami.discrete_outcomes()
        self.ordered_outcomes = sorted(
            [(self._utility_function(outcome), outcome) for outcome in outcomes],
            key=lambda x: x[0],
            reverse=True,
        )


    def is_betteroffer(self, utility):
        """Method to compute the N most likely future offers of the adversary and their utility for the agent
        Args:
            offer (Tuple[float]): offer of the opponent (p1, p2, ..., pn)
        Returns:
            decision (bool): whether or not we can expect a better offer
        """
        model = GaussianProcessRegressor()
        # check the size
        model.fit(range(self.t),self.past_offers)
        # The next offers predicted
        next_offers = model.predict(range(self.t,self.T))
        # Utilities of the next offers predicted
        next_utility = [self.ufun[offer] for offer in next_offers]
        if utility <= max(next_utility):
            return True
        return False


    def adversary_model_utility(self, offer):
        """ Method to compute the theoretical utility of the adversary for a given offer
        Args:
            offer (Tuple[float]): offer of the opponent (p1, p2, ..., pn)
        Returns:
            utility (float): utility value of the opponent
        """
        nb_iter = min(self.t, 100)
        if self.t<=1:
            return 10.
        utility = 0.
        for i in range(1,nb_iter+1):
            N_i = np.linalg.norm(np.array(offer)-np.array(self.past_offers[-i]))
            M_i = i
            P_i = 1/((N_i*M_i)+1e-6)
            utility += (10-3*((self.t+i)/10000))*P_i
        if utility > 50: # unexpected explosion of the value
            return 5.
        return utility


    def nash(self, offer):
        """Method to compute the Nash value of an offer
        Args:
            offer (Tuple[float]): offer of the opponent (p1, p2, ..., pn)
        Returns:
            nash_value (float): nash value of an offer
            adversary_utility (float): utility value of the opponent
            agent_utility (float): utility value of the agent
        """
        # adversary
        adversary_utility = min(self.adversary_model_utility(offer),10)
        # agent
        agent_utility = self.ufun[offer]
        # computing nash
        nash_value = np.sqrt(adversary_utility*agent_utility)
        return nash_value, adversary_utility, agent_utility


    def calc_nash_point(self):
        """ Method to compute the Nash equilibrium
        Returns:
            u_adv (float): adversary utility for the adversary
            u_ag (float): utility of the agent for the offer of nash equilibrium
        """
        outcomes = self._ami.discrete_outcomes()
        # We find the offer maximising the Nash equilibrium, we keep the relative utilities
        nash_value = 0.
        u_adv = 0.
        u_ag = 0.
        for offer in outcomes:
            n, u_adv_loc, u_ag_loc = self.nash(offer)
            if n>nash_value :
                nash_value = n
                u_adv = u_adv_loc
                u_ag = u_ag_loc
        print(self.t,nash_value,u_adv, u_ag)
        self.nash_point = [u_adv, u_ag]


    def respond(self, state: MechanismState, offer: "Outcome") -> "ResponseType":  # called when the agent receives the offer "offer";
        """ Returns either ACCEPT_OFFER or REJECT_OFFER. You also can access the offers
        Args:
            offer (Tuple[float]): offer of the opponent (p1, p2, ..., pn)
        Returns:
            ResponseType.ACCEPT_OFFER (ResponseType): Accept offer
            ResponseType.REJECT_OFFER (ResponseType): Reject offer
        """
        self.past_offers.append(list(offer))
        self.t +=1
        if self.upper_u_threshold is None:
            outcomes = self._ami.discrete_outcomes()
            utilities = [self.ufun[outcome] for outcome in outcomes]
            self.max_utility =np.max(utilities)
            self.upper_u_threshold = self.max_utility*0.92
            self.lower_u_threshold = self.max_utility*0.3
        utility = self.ufun[offer]
        self.calc_nash_point()

        if utility > self.upper_u_threshold:
            print("Accept proposition close to the upper threshold")
            return ResponseType.ACCEPT_OFFER

        elif self.t > self.T and utility > self.lower_u_threshold:
            print("Accept the lower threshold")
            return ResponseType.ACCEPT_OFFER

        elif utility > self.nash_point[1] and self.t>20:
            print("Accept because the propositon is better than the computed nash point",self.nash_point)
            return ResponseType.ACCEPT_OFFER

        elif self.t > int(self.T*0.6) and not self.is_betteroffer(utility):
            print("Accept because there won't be any better offer")
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER


    def propose(self, state: MechanismState) -> Optional["Outcome"]:
        """ Method called when the agent needs to make an offer on 1st round or later. Should send a proposal.
        Args:
            state (MechanismState): state of the negotiation
        Returns:
            best_offer (Tuple[float]): best offer
            max_offer (Tuple[float]): max offer
        """
        # Computing our goal utility
        outcomes = self._ami.discrete_outcomes()
        if self.t > 0:
            last_offer = self.past_offers[-1]
            # The best offer calculated thanks to nash
            best_offer = None
        # Offer maximizing our utility
        max_offer = outcomes[0]
        max_utility = self.ufun[max_offer]
        if self.t > 0 :
            # Distance to the goal we accept
            d=1
            max_adversary_utilities = np.max([self.adversary_model_utility(offer) for offer in outcomes])
            last_adversary_utility = self.adversary_model_utility(last_offer)
            # Concession of the adversary
            concession = np.abs((max_adversary_utilities-last_adversary_utility)/(max_adversary_utilities-self.nash_point[0] +1e-6))

            # We look for a proposal for which our utility makes a concession equal to the one of the adversary
            obj = self.max_utility-(self.max_utility-self.nash_point[1])*concession
        
        # Computing the best solution, which is the utility near our goal and which maximize the adversarial utility
        for offer in outcomes:
            u = self.ufun[offer]
            # If the offer is at the distance d of our goal utility
            if self.t > 0 and abs(u-obj) < d:
                if best_offer == None :
                    best_offer = offer
                    best_adversarial_utility = self.adversary_model_utility(offer)
                elif self.adversary_model_utility(offer) > best_adversarial_utility :
                    best_offer = offer
                    best_adversarial_utility = self.adversary_model_utility(offer)
            if u > max_utility:
                max_offer = offer
                max_utility = self.ufun[max_offer]

        # At the beginning of the negociation or if there is no solution, we return the offer that maximize our utility
        if self.t < 10 or best_offer == None:
            return max_offer

        return best_offer

############################

def build_utility(outcomes, points):
    """ Function to build utilities over 3-dimensional vectors.
    Args:
        outcomes (List[Tuple[Float]]): List of possible outcomes
        points (Dict[Float, List[Tuple[Float]]]): Dictionary of possible offers

    Returns:
        utilities (Array[Float]): Utility values for all possible outcomes
    """
    utilities = []
    for o in outcomes:
        p1 = min(points.keys(), key=lambda p: np.linalg.norm(np.array(p) - np.array(o)))
        p2 = min([k for k in points.keys() if k != p1], key=lambda p: np.linalg.norm(np.array(p) - np.array(o)))
        p3 = min([k for k in points.keys() if k != p1 and k != p2], key=lambda p: np.linalg.norm(np.array(p) - np.array(o)))
        utilities.append((points[p1]/max(1., np.linalg.norm(np.array(p1) - np.array(o))) + points[p2]/np.linalg.norm(np.array(p2) - np.array(o)) +
                         points[p3]/np.linalg.norm(np.array(p3) - np.array(o))) / (1/max(1., np.linalg.norm(np.array(p1) - np.array(o))) +
                         1/np.linalg.norm(np.array(p2) - np.array(o)) + 1/np.linalg.norm(np.array(p3) - np.array(o))))
    return np.array(utilities)


if __name__ == '__main__':
    n_steps=100000
    # First, choose your agents. There are several in negmas.sao.negotiators. Check it out and test VS your own!a1 = NaiveTitForTatNegotiator()
    a1 = CustomNegotiator(n_steps)
    # a1 = NaiveTitForTatNegotiator()
    a2 = CustomNegotiator(n_steps)
    # Create the possible outcomes of the negotiation
    outcomes = [(a, b, c, d, e) for a in range(10) for b in range(10) for c in range(10) for d in range(10) for e in range(10)]
    print("outcome OK")
    time_0 = time.time()
    # create the utility corresponding to the outcomes
    points = {}
    for i in range(50):
        points[(int(random.random()*10), int(random.random()*10), int(random.random()*10),
                int(random.random()*10), int(random.random()*10))] = int(random.random() * 10)
    u1 = build_utility(outcomes, points)
    for i in range(50):
        points[(int(random.random()*10), int(random.random()*10), int(random.random()*10),
                int(random.random()*10), int(random.random()*10))] = int(random.random() * 10)
    u2 = build_utility(outcomes, points)
    print("utility OK")
    # The protocole, 100000 steps to find an agreement
    neg = SAOMechanism(outcomes=outcomes, n_steps=n_steps, outcome_type=tuple)
    # Allocate the preference profiles to the agents
    neg.add(a1, ufun=u1)
    neg.add(a2, ufun=u2)
    # And launch negotiation
    neg.run()
    # Print offers & utility
    a1offers = neg.negotiator_offers(a1.id)
    a2offers = neg.negotiator_offers(a2.id)
    print("\n=========================== Offers ===========================")
    print("Agent 1 ({}): ".format(a1.__class__.__name__) + str(a1offers))
    print("Agent 2 ({}): ".format(a2.__class__.__name__) + str(a2offers))
    print("\n=========================== Agreement ===========================")
    print(neg.agreement)
    print("\n=========================== Utilities ===========================")
    utility_a1 = a1.utility_function.mapping[neg.agreement] if neg.agreement else 0
    utility_a2 = a2.utility_function.mapping[neg.agreement] if neg.agreement else 0
    print("Utility of Agent 1 ({}): ".format(a1.__class__.__name__) + str(utility_a1))
    print("Utility of Agent 2 ({}): ".format(a2.__class__.__name__) + str(utility_a2))
    u1 = u1.reshape(100,100,10)
    print(time.time() - time_0)
    plt.imshow(u1[:,:,0])
    plt.show()
