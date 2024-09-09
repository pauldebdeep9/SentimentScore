This paper proposes a robust optimization method for procurement portfolios. The proposal is based on mixed integer linear programming (MILP) and considers the lead time and the costs associated with each factory. To address the uncertainty of future forecasts, some deterministic variables are replaced with sampled data, which is a common practice in robust optimization. Experimental results show that the proposed method outperforms both the baseline method and the deterministic method.


I have some comments and questions regarding the abstract. In my opinion, the abstract should provide the overview of your proposal to clarify your contributions.
- The abstract should clearly state the meanings of “Risk-Aware,” “Multiperiod,” and “Uncertainty” as used in the title. “Risk-aware” and “Uncertainty” seem to have the duplicated meanings, both indicating the uncertainty of future forecasts. Does “Risk-aware” in the title refer to robust optimization as used in the abstract?

- Although the title contains the word of “Complexity,” I cannot identify your contribution to addressing the “Complexity” after reading the main document. “Complexity” seems to refer to the situation where the optimization problem should consider the lead time and the costs associated with each factory. Is it particularly complex for the MILP framework? If so, the main document should clarify the differences of the “Complexity” between your problem setting and a typical MILP problem setting, as well as how your proposal overcomes this complexity.

- The abstract should clearly state the meaning of “Extreme.” Because this is a technical report, such an adjective should be mathematically and quantitatively defined. Can you distinguish “EXTREME Complexity and Uncertainty” from merely “Complexity and Uncertainty”? To solve such an “Extreme” problem, what contributions does your research make?


I have one question about your experimental results. What causes the difference in the performance between the in-sample and out-sample experiments of your proposal, as described in Fig. 5? Equations (1) and (12) are optimization methods and seem to have no trainable parameters that can be used in the out-of-sample experiment.


The following are small comments regarding the writing.
- The lower left of the first page should include the affiliations of the authors and the email address of the corresponding author. Please refer to the template.

- In my opinion, Sections 2.4 and 2.5 should be included to a new section, such as “3. Results & Discussion.”

- The terms “insample” and “outsample” should be replaced with “in-sample” and “out-sample,” respectively.

- Some sentences are not easy to read due to a lack of commas; for example, the first sentence of Section 2.4.1.



