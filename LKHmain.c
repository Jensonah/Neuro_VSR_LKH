#include "LKH.h"
#include "Genetic.h"
#include <assert.h>


// Jens' function
float bound(float number){

    float max = 10000;
    float min = -10000;

    if(number > max)
        return max;
    if(number < min)
        return min;
    return number;
}


// Jens' function
void output_q_values()
{
    // Make filepath (for file descriptor)
    char filepath[100] = "input_nn/";
    strcat(filepath, outName);
    strcat(filepath, "_data.csv");

    // Open file
    FILE *fd;
    fd = fopen(filepath, "a");
    if (!fd)
    {
        perror("fopen");
    }

    if(ftell(fd) == 0)
        fprintf(fd, "Cost,Alpha,Q-Value\n");

    for (int i = 1; i <= Dimension; i++) 
    {
        int count = 0;
         // For all nodes in the candidate set of current node
        for (Candidate* CNN = NodeSet[i].CandidateSet; CNN->To; CNN++) 
        {
            // Print their Cost, Alpha and A-value
            fprintf(fd, "%d,%d,%f\n", CNN->Cost, CNN->Alpha, bound(CNN->Value));
            count += 1;
        }
        // print amount of candidates
        fprintf(fd, "%d\n", count);
    }

    fclose(fd);
}


// Jens' function
void read_model_predictions(char *outName)
{

    char filepath[100] = "out_nn/";
    strcat(filepath, outName);
    strcat(filepath, "_data.csv");

    // Open file
    FILE *fd;
    fd = fopen(filepath, "r");
    if (!fd)
    {
        perror("fopen");
    }

    for(int i = 0; i < Dimension; i++)
    {
        if(!fscanf(fd, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf", &Q_values[i*8], &Q_values[i*8 + 1], &Q_values[i*8 + 2], &Q_values[i*8 + 3], &Q_values[i*8 + 4], &Q_values[i*8 + 5], &Q_values[i*8 + 6], &Q_values[i*8 + 7]))
        {
            printf("scanf failed!\n");
            assert(0);
        }
    }
    fclose(fd);
}


/*
 * This file contains the main function of the program.
 */

int main(int argc, char *argv[])
{

    GainType Cost, OldOptimum;
    double Time, LastTime;
    
    /* Read the specification of the problem */
    if (argc >= 2)
    {
        ParameterFileName = argv[1];
    }
    
    if (argc >= 3)
    {    
        outName = argv[2];
    }

    ReadParameters();
    MaxMatrixDimension = 20000;
    MergeWithTour = Recombination == IPT ? MergeWithTourIPT :
        MergeWithTourGPX2;
    ReadProblem();
    LastTime = GetTime();

    if (SubproblemSize > 0) {
        if (DelaunayPartitioning)
            SolveDelaunaySubproblems();
        else if (KarpPartitioning)
            SolveKarpSubproblems();
        else if (KCenterPartitioning)
            SolveKCenterSubproblems();
        else if (KMeansPartitioning)
            SolveKMeansSubproblems();
        else if (RohePartitioning)
            SolveRoheSubproblems();
        else if (MoorePartitioning || SierpinskiPartitioning)
            SolveSFCSubproblems();
        else
            SolveTourSegmentSubproblems();
        return EXIT_SUCCESS;
    }
    AllocateStructures();
    CreateCandidateSet();
    InitializeStatistics();

    if (Norm != 0)
        BestCost = PLUS_INFINITY;
    else {
        /* The ascent has solved the problem! */
        Optimum = BestCost = (GainType) LowerBound;
        UpdateStatistics(Optimum, GetTime() - LastTime);
        RecordBetterTour();
        RecordBestTour();
        WriteTour(OutputTourFileName, BestTour, BestCost);
        WriteTour(TourFileName, BestTour, BestCost);
        Runs = 0;
    }

    Q_values = malloc(sizeof(double)*Dimension*8);
    read_model_predictions(outName);
    

    // Here the starting values are set
    for (int i = 1; i <= Dimension; i++) {
        int j = 0;
        for (Candidate* CNN = NodeSet[i].CandidateSet; CNN->To; CNN++) {
            CNN->Value = Q_values[(i-1)*8 + j];
            j += 1;
            //CNN->Value = LB2 / ((double)(Distance(&NodeSet[i], CNN->To) + CNN->Alpha));
        }
    }

    //output_q_values(outName); // starting q values at this point
    //return EXIT_SUCCESS;

    Node* From = FirstNode;
    Candidate *NFrom, *NN, Temp;
    do{
        for (NFrom = From->CandidateSet; NFrom->To; NFrom++) {
            Temp = *NFrom;
            for (NN = NFrom - 1;
                 NN >= From->CandidateSet &&
                 (Temp.Value > NN->Value ||
                  (Temp.Value == NN->Value && Temp.Alpha < NN->Alpha)); NN--)
                *(NN + 1) = *NN;
            *(NN + 1) = Temp;
        }
    }
    while((From = From->Suc) != FirstNode);
    
    /* Find a specified number (Runs) of local optima */
    for (Run = 1; Run <= Runs; Run++) { 
        LastTime = GetTime();
        Cost = FindTour();      /* using the Lin-Kernighan heuristic */
        
        if (MaxPopulationSize > 1) {
            /* Genetic algorithm */
            int i;
            for (i = 0; i < PopulationSize; i++) {
                GainType OldCost = Cost;
                Cost = MergeTourWithIndividual(i);
                if (TraceLevel >= 1 && Cost < OldCost) {
                    printff("  Merged with %d: Cost = " GainFormat, i + 1,
                            Cost);
                    if (Optimum != MINUS_INFINITY && Optimum != 0)
                        printff(", Gap = %0.4f%%",
                                100.0 * (Cost - Optimum) / Optimum);
                    printff("\n");
                }
            }
            if (!HasFitness(Cost)) {
                if (PopulationSize < MaxPopulationSize) {
                    AddToPopulation(Cost);
                    if (TraceLevel >= 1)
                        PrintPopulation();
                } else if (Cost < Fitness[PopulationSize - 1]) {
                    i = ReplacementIndividual(Cost);
                    ReplaceIndividualWithTour(i, Cost);
                    if (TraceLevel >= 1)
                        PrintPopulation();
                }
            }
        } else if (Run > 1)
            Cost = MergeTourWithBestTour();
        if (Cost < BestCost) {
            BestCost = Cost;
            RecordBetterTour();
            RecordBestTour();
            WriteTour(OutputTourFileName, BestTour, BestCost);
            WriteTour(TourFileName, BestTour, BestCost);
        }
        OldOptimum = Optimum;
        if (Cost < Optimum) {
            if (FirstNode->InputSuc) {
                Node *N = FirstNode;
                while ((N = N->InputSuc = N->Suc) != FirstNode);
            }
            Optimum = Cost;
            printff("*** New optimum = " GainFormat " ***\n\n", Optimum);
        }
        Time = fabs(GetTime() - LastTime);
        UpdateStatistics(Cost, Time);
        if (TraceLevel >= 1 && Cost != PLUS_INFINITY) {
            printff("Run %d: Cost = " GainFormat, Run, Cost);
            if (Optimum != MINUS_INFINITY && Optimum != 0)
                printff(", Gap = %0.4f%%",
                        100.0 * (Cost - Optimum) / Optimum);
            printff(", Time = %0.2f sec. %s\n\n", Time,
                    Cost < Optimum ? "<" : Cost == Optimum ? "=" : "");
        }
        if (StopAtOptimum && Cost == OldOptimum && MaxPopulationSize >= 1) {
            Runs = Run;
            break;
        }
        if (PopulationSize >= 2 &&
            (PopulationSize == MaxPopulationSize ||
             Run >= 2 * MaxPopulationSize) && Run < Runs) {
            Node *N;
            int Parent1, Parent2;
            Parent1 = LinearSelection(PopulationSize, 1.25);
            do
                Parent2 = LinearSelection(PopulationSize, 1.25);
            while (Parent2 == Parent1);
            ApplyCrossover(Parent1, Parent2);
            N = FirstNode;
            do {
                if (ProblemType != HCP && ProblemType != HPP) {
                    int d = C(N, N->Suc);
                    AddCandidate(N, N->Suc, d, INT_MAX);
                    AddCandidate(N->Suc, N, d, INT_MAX);
                }
                N = N->InitialSuc = N->Suc;
            }
            while (N != FirstNode);
        }
        SRandom(++Seed);
    }

    free(Q_values);

    PrintStatistics();
	// system("pause");
    return EXIT_SUCCESS;
}
