:- dynamic([item/2,depPath_fl/2,maxDepPath/2]).

type(train).
%type(test).
%type(augmented).

depTree:-
    retractall(item(_,_)),
    retractall(depPath(_,_)),
    retractall(maxDepPath(_,_)),
    read_data,
    generate_features,
    (
        (type(train), open('../data/prologData/depTree_features_train.txt',write,Out))
        ;
        (type(test), open('../data/prologData/depTree_features_test.txt',write,Out) )
        ;
        (type(augmented), open('../data/prologData/depTree_features_augmented.txt',write,Out))
    ),
    output_file(Out),
    close(Out).


output_file(Out):-
    retract(depPath_fl(SentID,L)),!,
    findall(FL,depPath_fl(SentID,FL), FeatList),
    list_append(FeatList,L,AllList),
    retractall(depPath_fl(SentID,_)),
    write(Out,SentID),
    write(Out,";"),
    write_out(Out,AllList),
    nl(Out),
    output_file(Out).

output_file(_).

list_append([],R,R).
list_append([H|T],Acc,R):-
     append(H,Acc,NewAcc),
     list_append(T,NewAcc,R).

write_out(Out,[[]|T]):- write_out(Out,T).
write_out(Out,[H]):-
    atomics_to_string(H,S),
    write(Out,S).
write_out(Out,[H|T]):-
    atomics_to_string(H,S),
    write(Out,S),write(Out,", "),
    write_out(Out,T).


read_data:-
    type(train),
    consult("../data/prologData/train_prolog_deptree.pl").

read_data:-
    type(test),
    consult("../data/prologData/test_prolog_deptree.pl").

read_data:-
    type(augmented),
    consult("../data/prologData/augmented_prolog_deptree.pl").

generate_features:-
    retract(item(SentID,token(StartID,_,'ROOT',_,_,_))),
    retractall(item(SentID,token(StartID,_,'ROOT',_,_,_))),
    generate_features(SentID,StartID),
    fail.
    
generate_features.



generate_features(SentID,ID):-
    % all paths
    findall(P,depPath(SentID,ID,B,P),L),
    % all maximal paths
    findall(P,(maxDepPath(SentID,ID,B,P),not(arc(B,_))),LMax),
    append(L,LMax,LResult),
    assert(depPath_fl(SentID,LResult)).




%token(ParID,ParWord,Dep,ChildID,ChildWord,ChildPOS)
    
depPath(_,B,B,[]).
depPath(SentID,A,B,[Dep,'#'|Path]):-
    item(SentID,token(A,_,Dep,C,_,_)),
    depPath(SentID,C,B,Path).

maxDepPath(SentID,B,B,[POS]):-
    not(item(SentID,token(B,_,_,_,_,_))),
    item(SentID,token(_,_,_,B,_,POS)).

maxDepPath(SentID,A,B,[Dep,'#'|Path]):-
    item(SentID,token(A,_,Dep,C,_,_)),
    maxDepPath(SentID,C,B,Path).



arc(a,b).
arc(a,c).
arc(b,c).
arc(d,e).
arc(a,d).



path(B,B,[B]).
path(A,B,[A|Path]):-
    arc(A,C),
    path(C,B,Path).

