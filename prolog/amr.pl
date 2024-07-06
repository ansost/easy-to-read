%type(train).
%type(test).
%type(augmented).
type(eval).

% collect all indices
transform_all:-
    retractall(features(_,_)),
    type(T),
    read_amr(T),
    findall(N,ex(N,_),L),
    transform_all(L),
    (
        (type(train), open('../data/prologData/features_train.txt',write,Out))
        ;
        (type(test), open('../data/prologData/features_test.txt',write,Out))
        ;
        (type(augmented), open('../data/prologData/features_augmented.txt',write,Out))
        ;
        (type(eval), open('../data/prologData/features_eval.txt',write,Out))
    ),
    output_file(Out),
    close(Out).

read_amr(train):-
    consult("../data/prologData/train_for_prolog_amr.pl").

read_amr(test):-
    consult("../data/prologData/test_for_prolog_amr.pl").

read_amr(augmented):-
    consult("../data/prologData/augmented_for_prolog_amr.pl").

read_amr(eval):-
    consult("../data/prologData/eval_for_prolog_amr.pl").

output_file(Out):-
    retract(features(N,L)),
    write(Out,N),
    write(Out,";"),
    write_features(Out,L),
    nl(Out),
    fail.

output_file(_).

% writes attributes paths as strings (including all prefixes)
write_features(_,[]).
write_features(Out,[(Path,V)|T]):-
     write_path(Out,Path,[],V),
     write_features(Out,T).

write_path(Out,[],Acc,V):-
     write_list(Out,Acc,V).
     
write_path(Out,[H|T],Acc,V):-
     append(Acc,[H,":"],AL),
     write_list(Out,AL),
     write_path(Out,T,AL,V).
     
write_list(Out,[]):- write(Out,", ").
write_list(Out,[H|T]):- write(Out,H),write_list(Out,T).


write_list(Out,[],V):- write(Out,V), write(Out,", ").
write_list(Out,[H|T],V):- write(Out,H),write_list(Out,T,V).


% transform each amr into feature(Index,FeatureList) entry
transform_all([]).
transform_all([H|T]):-
     transform(H),!,
     transform_all(T).


transform(N):-
    retractall(triples(_,_,_)),
    retractall(triples_fl(_,_,_)),
    ex(N,AMR),
    amr(AMR,Result), % transforms structure into clean nested list structure
    amr_to_triples(Result),!,  % extracts triples(type,attribute,value-type)
    %destroy_loops,
    Result = [Inst|_],
    feature_list(Inst), % combines triples to triples_fl(type,max_attribute_list, value)
    setof((Attr,V),Inst^triples_fl(Inst,Attr,V),List),
    assert(features(N,List)).

/*
destroy_loops:-
    way(A,A,[A],W),
    reverse(W,[D,C|_]),
    retract(triples(C,_,D)),
    destroy_loops.

destroy_loops.
*/
    
amr(V, V):- not(is_list(V)).

amr([attr-A,V|T],[[attr-A,V_AMR]|T_AMR]):-
    !,
    amr(V,V_AMR),
    amr(T,T_AMR).
    


amr([H|T],[H|T_AMR]):-
    amr(T,T_AMR).

amr([],[]).


amr_to_triples([Inst|AV_list]):-
    amr_to_triples(Inst,AV_list).
    
amr_to_triples(_,[]).
amr_to_triples(Inst,[[A,V]|List]):-
     amr_to_triples(Inst,A,V),
     amr_to_triples(Inst,List).

     
amr_to_triples(Inst,A,[V_Inst|List]):-
     !,
     assert(triples(Inst,A,V_Inst)),
     amr_to_triples(V_Inst,List).
     
amr_to_triples(Inst,A,V):-
     not(is_list(V)),
     assert(triples(Inst,A,V)).


feature_list(Inst):-
     feature_list(Inst,[],Attr,V),
     assert(triples_fl(Inst,Attr,V)),
     fail.

feature_list(_).


feature_list(V,_Acc,[],V):-
     not(triples(V,_,_)).

feature_list(Inst,Acc,[A|Attr],V):-
     triples(Inst,A,V1),
     not(member(V1,Acc)),
     feature_list(V1,[V1|Acc],Attr,V).


feature_list(Inst,Acc,[A],"loop"):-
     triples(Inst,A,V1),
     member(V1,Acc).


