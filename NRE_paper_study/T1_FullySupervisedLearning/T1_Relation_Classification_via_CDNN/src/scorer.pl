#!/usr/bin/perl -w
#
#
#  Author: Preslav Nakov
#          nakov@comp.nus.edu.sg
#          National University of Singapore
#
#  WHAT: This is the official scorer for SemEval-2010 Task #8.
#
#
#  Last modified: March 22, 2010
#
#  Current version: 1.2
#
#  Revision history:
#    - Version 1.2 (fixed a bug in the precision for the scoring of (iii))
#    - Version 1.1 (fixed a bug in the calculation of accuracy)
#
#
#  Use:
#     semeval2010_task8_scorer-v1.1.pl <PROPOSED_ANSWERS> <ANSWER_KEY>
#
#  Example2:
#     semeval2010_task8_scorer-v1.1.pl proposed_answer1.txt answer_key1.txt > result_scores1.txt
#     semeval2010_task8_scorer-v1.1.pl proposed_answer2.txt answer_key2.txt > result_scores2.txt
#     semeval2010_task8_scorer-v1.1.pl proposed_answer3.txt answer_key3.txt > result_scores3.txt
#
#  Description:
#     The scorer takes as input a proposed classification file and an answer key file.
#     Both files should contain one prediction per line in the format "<SENT_ID>	<RELATION>"
#     with a TAB as a separator, e.g.,
#           1	Component-Whole(e2,e1)
#           2	Other
#           3	Instrument-Agency(e2,e1)
#               ...
#     The files do not have to be sorted in any way and the first file can have predictions
#     for a subset of the IDs in the second file only, e.g., because hard examples have been skipped.
#     Repetitions of IDs are not allowed in either of the files.
#
#     The scorer calculates and outputs the following statistics:
#        (1) confusion matrix, which shows
#           - the sums for each row/column: -SUM-
#           - the number of skipped examples: skip
#           - the number of examples with correct relation, but wrong directionality: xDIRx
#           - the number of examples in the answer key file: ACTUAL ( = -SUM- + skip + xDIRx )
#        (2) accuracy and coverage
#        (3) precision (P), recall (R), and F1-score for each relation
#        (4) micro-averaged P, R, F1, where the calculations ignore the Other category.
#        (5) macro-averaged P, R, F1, where the calculations ignore the Other category.
#
#     Note that in scores (4) and (5), skipped examples are equivalent to those classified as Other.
#     So are examples classified as relations that do not exist in the key file (which is probably not optimal).
#
#     The scoring is done three times:
#       (i)   as a (2*9+1)-way classification
#       (ii)  as a (9+1)-way classification, with directionality ignored
#       (iii) as a (9+1)-way classification, with directionality taken into account.
#     
#     The official score is the macro-averaged F1-score for (iii).
#

use strict;


###############
###   I/O   ###
###############

if ($#ARGV != 1) {
	die "Usage:\nsemeval2010_task8_scorer.pl <PROPOSED_ANSWERS> <ANSWER_KEY>\n";
}

my $PROPOSED_ANSWERS_FILE_NAME = $ARGV[0];
my $ANSWER_KEYS_FILE_NAME      = $ARGV[1];


################
###   MAIN   ###
################

my (%confMatrix19way, %confMatrix10wayNoDir, %confMatrix10wayWithDir) = ();
my (%idsProposed, %idsAnswer) = ();
my (%allLabels19waylAnswer, %allLabels10wayAnswer) = ();
my (%allLabels19wayProposed, %allLabels10wayNoDirProposed, %allLabels10wayWithDirProposed) = ();

### 1. Read the file contents
my $totalProposed = &readFileIntoHash($PROPOSED_ANSWERS_FILE_NAME, \%idsProposed);
my $totalAnswer = &readFileIntoHash($ANSWER_KEYS_FILE_NAME, \%idsAnswer);

### 2. Calculate the confusion matrices
foreach my $id (keys %idsProposed) {

	### 2.1. Unexpected IDs are not allowed
	die "File $PROPOSED_ANSWERS_FILE_NAME contains a bad ID: '$id'"
		if (!defined($idsAnswer{$id}));

	### 2.2. Update the 19-way confusion matrix
	my $labelProposed = $idsProposed{$id};
	my $labelAnswer   = $idsAnswer{$id};
	$confMatrix19way{$labelProposed}{$labelAnswer}++;
	$allLabels19wayProposed{$labelProposed}++;

	### 2.3. Update the 10-way confusion matrix *without* direction
	my $labelProposedNoDir = $labelProposed;
	my $labelAnswerNoDir   = $labelAnswer;
	$labelProposedNoDir =~ s/\(e[12],e[12]\)[\n\r]*$//;
	$labelAnswerNoDir =~ s/\(e[12],e[12]\)[\n\r]*$//;
	$confMatrix10wayNoDir{$labelProposedNoDir}{$labelAnswerNoDir}++;
	$allLabels10wayNoDirProposed{$labelProposedNoDir}++;

	### 2.4. Update the 10-way confusion matrix *with* direction
	if ($labelProposed eq $labelAnswer) { ## both relation and direction match
		$confMatrix10wayWithDir{$labelProposedNoDir}{$labelAnswerNoDir}++;
		$allLabels10wayWithDirProposed{$labelProposedNoDir}++;
	}
	elsif ($labelProposedNoDir eq $labelAnswerNoDir) { ## the relations match, but the direction is wrong
		$confMatrix10wayWithDir{'WRONG_DIR'}{$labelAnswerNoDir}++;
		$allLabels10wayWithDirProposed{'WRONG_DIR'}++;
	}
	else { ### Wrong relation
		$confMatrix10wayWithDir{$labelProposedNoDir}{$labelAnswerNoDir}++;
		$allLabels10wayWithDirProposed{$labelProposedNoDir}++;
	}
}

### 3. Calculate the ground truth distributions
foreach my $id (keys %idsAnswer) {

	### 3.1. Update the 19-way answer distribution
	my $labelAnswer = $idsAnswer{$id};
	$allLabels19waylAnswer{$labelAnswer}++;

	### 3.2. Update the 10-way answer distribution
	my $labelAnswerNoDir = $labelAnswer;
	$labelAnswerNoDir =~ s/\(e[12],e[12]\)[\n\r]*$//;
	$allLabels10wayAnswer{$labelAnswerNoDir}++;
}

### 4. Check for proposed classes that are not contained in the answer key file: this may happen in cross-validation
foreach my $labelProposed (sort keys %allLabels19wayProposed) {
	if (!defined($allLabels19waylAnswer{$labelProposed})) {
		print "!!!WARNING!!! The proposed file contains $allLabels19wayProposed{$labelProposed} label(s) of type '$labelProposed', which is NOT present in the key file.\n\n";
	}
}

### 4. 19-way evaluation with directionality
print "<<< (2*9+1)-WAY EVALUATION (USING DIRECTIONALITY)>>>:\n\n";
&evaluate(\%confMatrix19way, \%allLabels19wayProposed, \%allLabels19waylAnswer, $totalProposed, $totalAnswer, 0);

### 5. Evaluate without directionality
print "<<< (9+1)-WAY EVALUATION IGNORING DIRECTIONALITY >>>:\n\n";
&evaluate(\%confMatrix10wayNoDir, \%allLabels10wayNoDirProposed, \%allLabels10wayAnswer, $totalProposed, $totalAnswer, 0);

### 6. Evaluate without directionality
print "<<< (9+1)-WAY EVALUATION TAKING DIRECTIONALITY INTO ACCOUNT -- OFFICIAL >>>:\n\n";
my $officialScore = &evaluate(\%confMatrix10wayWithDir, \%allLabels10wayWithDirProposed, \%allLabels10wayAnswer, $totalProposed, $totalAnswer, 1);

### 7. Output the official score
printf "<<< The official score is (9+1)-way evaluation with directionality taken into account: macro-averaged F1 = %0.2f%s >>>\n", $officialScore, '%';


################
###   SUBS   ###
################

sub getIDandLabel() {
	my $line = shift;
	return (-1,()) if ($line !~ /^([0-9]+)\t([^\r]+)\r?\n$/);

	my ($id, $label) = ($1, $2);

    return ($id, '_Other') if ($label eq 'Other');

	return ($id, $label)
    if (($label eq 'Cause-Effect(e1,e2)')       || ($label eq 'Cause-Effect(e2,e1)')       ||
		($label eq 'Component-Whole(e1,e2)')    || ($label eq 'Component-Whole(e2,e1)')    ||
		($label eq 'Content-Container(e1,e2)')  || ($label eq 'Content-Container(e2,e1)')  ||
		($label eq 'Entity-Destination(e1,e2)') || ($label eq 'Entity-Destination(e2,e1)') ||
		($label eq 'Entity-Origin(e1,e2)')      || ($label eq 'Entity-Origin(e2,e1)')      ||
		($label eq 'Instrument-Agency(e1,e2)')  || ($label eq 'Instrument-Agency(e2,e1)')  ||
		($label eq 'Member-Collection(e1,e2)')  || ($label eq 'Member-Collection(e2,e1)')  ||
		($label eq 'Message-Topic(e1,e2)')      || ($label eq 'Message-Topic(e2,e1)')      ||
		($label eq 'Product-Producer(e1,e2)')   || ($label eq 'Product-Producer(e2,e1)'));
	
	return (-1, ());
}


sub readFileIntoHash() {
	my ($fname, $ids) = @_;
	open(INPUT, $fname) or die "Failed to open $fname for text reading.\n";
	my $lineNo = 0;
	while (<INPUT>) {
		$lineNo++;
		my ($id, $label) = &getIDandLabel($_);
		die "Bad file format on line $lineNo: '$_'\n" if ($id < 0);
		if (defined $$ids{$id}) {
			s/[\n\r]*$//;
			die "Bad file format on line $lineNo (ID $id is already defined): '$_'\n";
		}
		$$ids{$id} = $label;
	}
	close(INPUT) or die "Failed to close $fname.\n";
	return $lineNo;
}


sub evaluate() {
	my ($confMatrix, $allLabelsProposed, $allLabelsAnswer, $totalProposed, $totalAnswer, $useWrongDir) = @_;

	### 0. Create a merged list for the confusion matrix
	my @allLabels = ();
	&mergeLabelLists($allLabelsAnswer, $allLabelsProposed, \@allLabels);

	### 1. Print the confusion matrix heading
	print "Confusion matrix:\n";
	print "       ";
	foreach my $label (@allLabels) {
		printf " %4s", &getShortRelName($label, $allLabelsAnswer);
	}
	print " <-- classified as\n";
	print "      +";
	foreach my $label (@allLabels) {
		print "-----";
	}
	if ($useWrongDir) {
		print "+ -SUM- xDIRx skip  ACTUAL\n";
	}
	else {
		print "+ -SUM- skip ACTUAL\n";
	}

	### 2. Print the rest of the confusion matrix
	my $freqCorrect = 0;
	my $ind = 1;
	my $otherSkipped = 0;
	foreach my $labelAnswer (sort keys %{$allLabelsAnswer}) {

		### 2.1. Output the short relation label
		printf " %4s |", &getShortRelName($labelAnswer, $allLabelsAnswer);

		### 2.2. Output a row of the confusion matrix
		my $sumProposed = 0;
		foreach my $labelProposed (@allLabels) {
			$$confMatrix{$labelProposed}{$labelAnswer} = 0
				if (!defined($$confMatrix{$labelProposed}{$labelAnswer}));
			printf "%4d ", $$confMatrix{$labelProposed}{$labelAnswer};
			$sumProposed += $$confMatrix{$labelProposed}{$labelAnswer};
		}

		### 2.3. Output the horizontal sums
		if ($useWrongDir) {
			my $ans = defined($$allLabelsAnswer{$labelAnswer}) ? $$allLabelsAnswer{$labelAnswer} : 0;
			$$confMatrix{'WRONG_DIR'}{$labelAnswer} = 0 if (!defined $$confMatrix{'WRONG_DIR'}{$labelAnswer});
			printf "| %4d  %4d  %4d %6d\n", $sumProposed, $$confMatrix{'WRONG_DIR'}{$labelAnswer}, $ans - $sumProposed - $$confMatrix{'WRONG_DIR'}{$labelAnswer}, $ans;
			if ($labelAnswer eq '_Other') {
				$otherSkipped = $ans - $sumProposed - $$confMatrix{'WRONG_DIR'}{$labelAnswer};
			}
		}
		else {
			my $ans = defined($$allLabelsAnswer{$labelAnswer}) ? $$allLabelsAnswer{$labelAnswer} : 0;
			printf "| %4d %4d %4d\n", $sumProposed, $ans - $sumProposed, $ans;
			if ($labelAnswer eq '_Other') {
				$otherSkipped = $ans - $sumProposed;
			}
		}

		$ind++;

		$$confMatrix{$labelAnswer}{$labelAnswer} = 0
			if (!defined($$confMatrix{$labelAnswer}{$labelAnswer}));
		$freqCorrect += $$confMatrix{$labelAnswer}{$labelAnswer};
	}
	print "      +";
	foreach (@allLabels) {
		print "-----";
	}
	print "+\n";
	
	### 3. Print the vertical sums
	print " -SUM- ";
	foreach my $labelProposed (@allLabels) {
		$$allLabelsProposed{$labelProposed} = 0
			if (!defined $$allLabelsProposed{$labelProposed});
		printf "%4d ", $$allLabelsProposed{$labelProposed};
	}
	if ($useWrongDir) {
		printf "  %4d  %4d  %4d %6d\n\n", $totalProposed - $$allLabelsProposed{'WRONG_DIR'}, $$allLabelsProposed{'WRONG_DIR'}, $totalAnswer - $totalProposed, $totalAnswer;
	}
	else {
		printf "  %4d %4d %4d\n\n", $totalProposed, $totalAnswer - $totalProposed, $totalAnswer;
	}

	### 4. Output the coverage
	my $coverage = 100.0 * $totalProposed / $totalAnswer;
	printf "%s%d%s%d%s%5.2f%s", 'Coverage = ', $totalProposed, '/', $totalAnswer, ' = ', $coverage, "\%\n";

	### 5. Output the accuracy
	my $accuracy = 100.0 * $freqCorrect / $totalProposed;
	printf "%s%d%s%d%s%5.2f%s", 'Accuracy (calculated for the above confusion matrix) = ', $freqCorrect, '/', $totalProposed, ' = ', $accuracy, "\%\n";

	### 6. Output the accuracy considering all skipped to be wrong
	$accuracy = 100.0 * $freqCorrect / $totalAnswer;
	printf "%s%d%s%d%s%5.2f%s", 'Accuracy (considering all skipped examples as Wrong) = ', $freqCorrect, '/', $totalAnswer, ' = ', $accuracy, "\%\n";

	### 7. Calculate accuracy with all skipped examples considered Other
	my $accuracyWithOther = 100.0 * ($freqCorrect + $otherSkipped) / $totalAnswer;
	printf "%s%d%s%d%s%5.2f%s", 'Accuracy (considering all skipped examples as Other) = ', ($freqCorrect + $otherSkipped), '/', $totalAnswer, ' = ', $accuracyWithOther, "\%\n";

	### 8. Output P, R, F1 for each relation
	my ($macroP, $macroR, $macroF1) = (0, 0, 0);
	my ($microCorrect, $microProposed, $microAnswer) = (0, 0, 0);
	print "\nResults for the individual relations:\n";
	foreach my $labelAnswer (sort keys %{$allLabelsAnswer}) {

		### 8.1. Consider all wrong directionalities as wrong classification decisions
		my $wrongDirectionCnt = 0;
		if ($useWrongDir && defined $$confMatrix{'WRONG_DIR'}{$labelAnswer}) {
			$wrongDirectionCnt = $$confMatrix{'WRONG_DIR'}{$labelAnswer};
		}

		### 8.2. Prevent Perl complains about unintialized values
		if (!defined($$allLabelsProposed{$labelAnswer})) {
			$$allLabelsProposed{$labelAnswer} = 0;
		}

		### 8.3. Calculate P/R/F1
		my $P  = (0 == $$allLabelsProposed{$labelAnswer}) ? 0
				: 100.0 * $$confMatrix{$labelAnswer}{$labelAnswer} / ($$allLabelsProposed{$labelAnswer} + $wrongDirectionCnt);
		my $R  = (0 == $$allLabelsAnswer{$labelAnswer}) ? 0
				: 100.0 * $$confMatrix{$labelAnswer}{$labelAnswer} / $$allLabelsAnswer{$labelAnswer};
		my $F1 = (0 == $P + $R) ? 0 : 2 * $P * $R / ($P + $R);

		### 8.4. Output P/R/F1
		if ($useWrongDir) {
			printf "%25s%s%4d%s(%4d +%4d)%s%6.2f", $labelAnswer,
				" :    P = ", $$confMatrix{$labelAnswer}{$labelAnswer}, '/', $$allLabelsProposed{$labelAnswer}, $wrongDirectionCnt, ' = ', $P;
		}
		else {
			printf "%25s%s%4d%s%4d%s%6.2f", $labelAnswer,
				" :    P = ", $$confMatrix{$labelAnswer}{$labelAnswer}, '/', ($$allLabelsProposed{$labelAnswer} + $wrongDirectionCnt), ' = ', $P;
		}
		printf"%s%4d%s%4d%s%6.2f%s%6.2f%s\n",
		  	 "%     R = ", $$confMatrix{$labelAnswer}{$labelAnswer}, '/', $$allLabelsAnswer{$labelAnswer},   ' = ', $R,
			 "%     F1 = ", $F1, '%';

		### 8.5. Accumulate statistics for micro/macro-averaging
		if ($labelAnswer ne '_Other') {
			$macroP  += $P;
			$macroR  += $R;
			$macroF1 += $F1;
			$microCorrect += $$confMatrix{$labelAnswer}{$labelAnswer};
			$microProposed += $$allLabelsProposed{$labelAnswer} + $wrongDirectionCnt;
			$microAnswer += $$allLabelsAnswer{$labelAnswer};
		}
	}

	### 9. Output the micro-averaged P, R, F1
	my $microP  = (0 == $microProposed)    ? 0 : 100.0 * $microCorrect / $microProposed;
	my $microR  = (0 == $microAnswer)      ? 0 : 100.0 * $microCorrect / $microAnswer;
	my $microF1 = (0 == $microP + $microR) ? 0 :   2.0 * $microP * $microR / ($microP + $microR);
	print "\nMicro-averaged result (excluding Other):\n";
	printf "%s%4d%s%4d%s%6.2f%s%4d%s%4d%s%6.2f%s%6.2f%s\n",
		      "P = ", $microCorrect, '/', $microProposed, ' = ', $microP,
		"%     R = ", $microCorrect, '/', $microAnswer, ' = ', $microR,
		"%     F1 = ", $microF1, '%';

	### 10. Output the macro-averaged P, R, F1
	my $distinctLabelsCnt = keys %{$allLabelsAnswer}; 
	## -1, if '_Other' exists
	$distinctLabelsCnt-- if (defined $$allLabelsAnswer{'_Other'});

	$macroP  /= $distinctLabelsCnt; # first divide by the number of non-Other categories
	$macroR  /= $distinctLabelsCnt;
	$macroF1 /= $distinctLabelsCnt;
	print "\nMACRO-averaged result (excluding Other):\n";
	printf "%s%6.2f%s%6.2f%s%6.2f%s\n\n\n\n", "P = ", $macroP, "%\tR = ", $macroR, "%\tF1 = ", $macroF1, '%';

	### 11. Return the official score
	return $macroF1;
}


sub getShortRelName() {
	my ($relName, $hashToCheck) = @_;
	return '_O_' if ($relName eq '_Other');
	die "relName='$relName'" if ($relName !~ /^(.)[^\-]+\-(.)/);
	my $result = (defined $$hashToCheck{$relName}) ? "$1\-$2" : "*$1$2";
	if ($relName =~ /\(e([12])/) {
		$result .= $1;
	}
	return $result;
}

sub mergeLabelLists() {
	my ($hash1, $hash2, $mergedList) = @_;
	foreach my $key (sort keys %{$hash1}) {
		push @{$mergedList}, $key if ($key ne 'WRONG_DIR');
	}
	foreach my $key (sort keys %{$hash2}) {
		push @{$mergedList}, $key if (($key ne 'WRONG_DIR') && !defined($$hash1{$key}));
	}
}
