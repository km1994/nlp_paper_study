#!/usr/bin/perl
#
# Usage: 
# 
#   correct-output.pl file
#
# 

=head1 $0

=head1 SYNOPSIS

 correct-output.pl file

 Examples:

   $ ./correct-output.pl file 

 Author: Aitor Gonzalez-Agirre

 Dec. 27, 2012

=cut
 
use warnings;
use strict;
use Scalar::Util qw(looks_like_number);

my $errors = 0;

open(I,$ARGV[0]) or die "Cannot open $ARGV[0]: $!";
while (<I>) {
    chomp ;
    if (/^\s*$/) { 
	warn "$ARGV[0]: empty line";
	$errors++;
    }
    else {
	my @sysvalues = split(/\t/,$_) ;

	if ((scalar(@sysvalues) != 1 ) and (scalar(@sysvalues) != 2)) {
	    warn "$ARGV[0]: number of values is not 1 or 2";
	    $errors++;
	    next ;
	}

	if ((!looks_like_number($sysvalues[0])) or ($sysvalues[0] eq "NaN")) {
	    warn "$ARGV[0]: score is not a number";
	    $errors++;
	    next ;
	}
	elsif (($sysvalues[0]<0) or ($sysvalues[0]>5)) {
	    warn "$ARGV[0]: score is not between 0 and 5";
	    $errors++;
	    next ;
	}
	
	next if (scalar(@sysvalues) == 1) ;

	if ((!looks_like_number($sysvalues[1])) or ($sysvalues[1] eq "NaN")) {
	    warn "$ARGV[0]: confidence score is not a number";
	    $errors++;
	    next ;
	}
	elsif (($sysvalues[1]<0) or ($sysvalues[1]>100)) {
	    warn "$ARGV[0]: confidence score is not between 0 and 100";
	    $errors++;
	    next ;
	}
    }
} 
close(I) ;

if ($errors == 0){
    print "Output file is OK!\n";
    exit(0);
}
else {
    print "Output file has $errors errors. Please correct them before sending.\n";
    exit(1);
}
