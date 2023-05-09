let _ =
  if Array.length Sys.argv < 3 then
    failwith "Usage: ./main <numpy_primitives.txt> <numpy expression>"
  else
    begin
    let fun_str = Sys.argv.(2) in
    let lexbuf = Lexing.from_string fun_str in
    Parser.function_expr Lexer.token lexbuf;
    end
