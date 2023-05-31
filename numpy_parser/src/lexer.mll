{
  open Parser

  module S = Set.Make(String)

  let rec read chan s =
    match input_line chan with
    | line -> read chan (S.add line s)
    | exception End_of_file -> close_in chan; s

  let in_channel = open_in Sys.argv.(1)

  let s = read in_channel S.empty

  let check_numpy_func fn = S.mem fn s

  let fail s =
    failwith (Printf.sprintf "Unexpected token: %s" s)

}

let digit  = ['0'-'9']
let number = digit+
let alpha  = ['a'-'z' 'A'-'Z']
let space  = [' ' '\t' '\r']

let float   = number ('.' number)?
let cst     = float | "np.pi" | "np.e" | "10e" '-'? number
let var     = "x"
let np_func = "np." alpha+
let param   = "axis" | "ord" | "p"


rule token = parse
  | space+             { token lexbuf }
  | "+"                { BINOP }
  | "-"                { MINUS }
  | "*"                { BINOP }
  | "/"                { BINOP }
  | "**"               { BINOP }
  | ","                { COMMA }
  | "("                { LPAR }
  | "["                { LBRACE }
  | ")"                { RPAR }
  | "]"                { RBRACE }
  | "="                { EQUAL }
  | cst                { CST }
  | var                { VAR }
  | np_func as np_func { if check_numpy_func np_func then NUMPY_FUNC else fail np_func }
  | param              { PARAM }
  | eof                { EOF }
  | _ as c             { fail String.(make 1 c)}
