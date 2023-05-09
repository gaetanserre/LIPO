{
  open Parser

  let check_numpy_func =
    let numpy_hash = Hashtbl.create 10 in

    let in_channel = open_in Sys.argv.(1) in
    let numpy_primitives = ref [] in

    try
      while true do
        numpy_primitives := input_line in_channel :: !numpy_primitives
      done;
      assert false 
    with e ->
      match e with
      | End_of_file -> (
        close_in in_channel;
        List.iter (fun (math_fun) -> Hashtbl.add numpy_hash math_fun true) !numpy_primitives;
        fun func_name -> Hashtbl.mem numpy_hash func_name
      )
      | _ -> (
        close_in_noerr in_channel;
        raise e
      )
     
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